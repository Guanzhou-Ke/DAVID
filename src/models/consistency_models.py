import torch
from torch import nn
from torch.nn import functional as F
import lightning.pytorch as pl
from torch.optim import lr_scheduler

from models.backbones import build_mlp, build_off_the_shelf_cnn
from losses import NeighborsLoss
from optimizer import get_optimizer
from utils import clustering_by_representation


class ConsistencyEncoder(pl.LightningModule):
    
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.save_hyperparameters()
        self.nheads = args.clustering.nheads
        self.pooling_method = args.fusion.pooling_method
        if args.backbone.type == 'cnn':
            self.enc = build_off_the_shelf_cnn(name=args.consis_enc.backbone,
                                               pretrained=False, 
                                               in_channel=args.consis_enc.in_channel)
        elif args.backbone.type == 'mlp':
            self.enc = build_mlp(layers=args.consis_enc.backbone,
                                 activation=args.consis_enc.activation,
                                 first_norm=args.consis_enc.first_norm)
        else:
            raise ValueError('Backbone type error.')
        assert(isinstance(self.nheads, int))
        assert(self.nheads > 0)
        self.cluster_head = nn.ModuleList([nn.Linear(args.hidden_dim, args.dataset.class_num) for _ in range(self.nheads)])
        self.criterion = NeighborsLoss(self.args.clustering.entropy_weight)

        # For evaluation
        self.targets = []
        self.reprs = []
        
        
    def configure_optimizers(self):
        # optimizer
        optimizer = get_optimizer(self.parameters(), self.args.train.lr, self.args.train.optim)
        if self.args.train.scheduler == 'constant':
            return optimizer
        elif self.args.train.scheduler == 'linear':
            lf = lambda x: (1 - x / self.args.train.epochs) * (1.0 - 0.1) + 0.1  # linear
            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
            return [optimizer], [scheduler]
        elif self.args.train.scheduler == 'consine':
            eta_min = self.args.train.lr * (self.args.train.lr_decay_rate ** 3)
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, self.args.train.epochs, eta_min=eta_min)
            return [optimizer], [scheduler]
        else:
            raise ValueError('Training scheduler value error.')
    
    
    def training_step(self, batch, batch_idx):
        anchors, neighbors, *_ = batch
        anchors, anchors_output = self(anchors, train=True)
        neighbors, neighbors_output = self(neighbors, train=True)
        total_loss, consistency_loss, entropy_loss = 0., 0., 0.
        for anchors_output_subhead, neighbors_output_subhead in zip(anchors_output, neighbors_output):
            sub_loss_, sub_consistency_loss_, sub_entropy_loss_ = self.criterion(anchors_output_subhead,
                                                                        neighbors_output_subhead)
            total_loss += sub_loss_
            consistency_loss += sub_consistency_loss_.item()
            entropy_loss += sub_entropy_loss_.item()
        
        self.log('train_loss', total_loss.item())
        self.log('consistency-loss', consistency_loss)
        self.log('entropy-loss', entropy_loss)
        
        optimizer = self.optimizers()
        lr = optimizer.param_groups[0]["lr"]
        self.log("lr", lr, prog_bar=True)
        
        return total_loss
    
    
    def validation_step(self, batch, batch_idx):
        anchors, *_, target = batch
        repr_ = self.consistency_repr(anchors)
        self.targets.append(target.detach().cpu())
        self.reprs.append(repr_.detach().cpu())
        

    def on_validation_epoch_end(self) -> None:
        if self.targets:
            targets = torch.concat(self.targets, dim=-1).numpy()
            reprs = torch.vstack(self.reprs).squeeze().numpy()
            acc, nmi, ari, class_acc, p, fscore = clustering_by_representation(reprs, targets)
            sync_dist = True
            self.log('clu-acc', acc, sync_dist=sync_dist, prog_bar=True)
            self.log('nmi', nmi, sync_dist=sync_dist, prog_bar=True)
            self.log('ari', ari, sync_dist=sync_dist, prog_bar=True)
            self.log('cls-acc', class_acc, sync_dist=sync_dist, prog_bar=True)
            self.log('p', p, sync_dist=sync_dist, prog_bar=True)
            self.log('fscore', fscore, sync_dist=sync_dist, prog_bar=True)
            self.targets.clear()
            self.reprs.clear()
        
    
    def load_pretext_encoder(self, model_state):
        self.enc.load_state_dict(model_state, strict=False)
  
        
    def forward(self, x, train=True):
        features = self.consistency_repr(x)
        out = [cluster_head(features) for cluster_head in self.cluster_head]
        
        if train:
            return features, out
        else:
            return features, out[0]
       
    
    def consistency_repr(self, Xs, normlization=False):
        features = [self.enc(x) for x in Xs]
        
        if self.pooling_method == 'mean':
            # use mean pooling to get common feature
            common_z = torch.stack(features, dim=-1).mean(dim=-1)
        elif self.pooling_method == 'sum':
            common_z = torch.stack(features, dim=-1).sum(dim=-1)
        elif self.pooling_method == 'first':
            common_z = features[0]
        if normlization:
            return F.normalize(common_z, dim=-1)
        else:
            return common_z
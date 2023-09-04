import torch
from torch import nn
from torch.nn import functional as F
import lightning.pytorch as pl
from torch.optim import lr_scheduler

from .backbones import build_mlp, build_off_the_shelf_cnn
from losses.dclloss import DCLLoss, DCLWLoss
from optimizer import get_optimizer


class ContrastiveModel(pl.LightningModule):
    """
    Contrastive Model for pretext training.
    """

    def __init__(self, args, mode='train') -> None:
        super().__init__()
        self.args = args
        self.save_hyperparameters()
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
        
        if mode == 'train':
            self.K = self.args.train.knn_num 
            self.temperature = args.consis_enc.temperature
            self.C = args.dataset.class_num
            self.mem_features = []
            self.mem_targets = []
            
        # projection head.
        self.projection_head = nn.Sequential(nn.Linear(args.consis_enc.output_dim, args.consis_enc.output_dim, bias=False), 
                                             nn.BatchNorm1d(args.consis_enc.output_dim),
                                             nn.ReLU(inplace=True), 
                                             nn.Linear(args.consis_enc.output_dim, args.consis_enc.project_dim, bias=True))
        self.pooling_method = args.fusion.pooling_method
        # loss
        if args.consis_enc.loss_type == 'dcl':
            self.contrast_criterion = DCLLoss(args.consis_enc.temperature)
        elif args.consis_enc.loss_type == 'dclw':
            self.contrast_criterion = DCLWLoss(args.consis_enc.temperature)
        else:
            raise ValueError('Loss type must be `dcl` or `dclw`.')
        
       
    def forward(self, x):
        feature = self.enc(x)
        cont_out = self.projection_head(feature)
        return feature, F.normalize(cont_out, dim=-1)
    
    
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
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, self.args.train.epochs // 10, eta_min=eta_min)
            return [optimizer], [scheduler]
        else:
            raise ValueError('Training scheduler value error.')
    
    def training_step(self, batch, batch_idx):
        """
        Using infoNCE to extract consistency information.
        """
        Xs, y = batch
        loss = 0.
        views = len(Xs)
        for i in range(views):
            for j in range(i+1, views):
                x1, x2 = Xs[i], Xs[j]
                _, out1 = self(x1)
                _, out2 = self(x2)
        
                loss += (self.contrast_criterion(out1, out2) + self.contrast_criterion(out2, out1))
        # Normalize.
        loss /= views
        self.log("train_loss", loss.item(), prog_bar=True)
        
        optimizer = self.optimizers()
        lr = optimizer.param_groups[0]["lr"]
        self.log("lr", lr, prog_bar=True)
        
        with torch.no_grad():
            self.mem_features.append(self.consistency_repr(Xs, normlization=True).detach().cpu())
            self.mem_targets.append(y.detach().cpu())
        
        return loss
    

    
    def validation_step(self, batch, batch_idx):
        if len(self.mem_features) == 0:
            return None
        Xs, y = batch
        top1, top5 = self.evaluation(Xs, y)
        self.log('val_top@1', top1, prog_bar=True, sync_dist=True)
        self.log('val_top@5', top5, prog_bar=True, sync_dist=True)
        
        
    def on_train_start(self) -> None:
        self.mem_features.clear()
        self.mem_targets.clear()
    
    
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
        
           
    def weighted_knn(self, predictions):
        features = torch.cat(self.mem_features).to(predictions.device)
        targets = torch.cat(self.mem_targets).to(predictions.device)
        retrieval_one_hot = torch.zeros(self.K, self.C).to(predictions.device)
        
        batch_size = predictions.shape[0]
        
        similarity = torch.mm(predictions, features.t())
        distances, indices = similarity.topk(self.K, largest=True, sorted=True)
        candidates = targets.view(1, -1).expand(batch_size, -1)
        retrieved_neighbors = torch.gather(candidates, 1, indices)
        
        retrieval_one_hot.resize_(batch_size * self.K, self.C).zero_()
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
        
        distances_transform = distances.clone().div_(self.temperature).exp_()
        probs = torch.sum(torch.mul(retrieval_one_hot.view(batch_size, -1, self.C),
                distances_transform.view(batch_size, -1, 1)), 1)
        class_preds = probs.argsort(dim=1, descending=True)

        return class_preds
    
    @torch.no_grad() 
    def evaluation(self, Xs, target):
        output = self.consistency_repr(Xs, normlization=True)
        total_num = output.size(0)
        output = self.weighted_knn(output) 
        
        top1 = torch.sum((output[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float())
        top5 = torch.sum((output[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float())

        top1 = (top1 / total_num) * 100
        top5 = (top5 / total_num) * 100
        return top1, top5
import argparse
import os

import torch
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar
import lightning.pytorch as pl

from configs.basic_config import get_cfg
from models.consistency_models import ConsistencyEncoder
from utils import (get_predictions,
                   neighbors_evaluate,
                   hungarian_evaluate,)
from utils.datatool import (get_train_transformations,
                            get_val_transformations,
                            get_train_dataset,
                            get_val_dataset)


# https://pytorch.org/docs/stable/elastic/run.html
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', '-f', type=str, help='Config File')
    args = parser.parse_args()
    return args


def main():
    # Load arguments.
    args = parse_args()
    config = get_cfg(args.config_file)

    pretext_dir = os.path.join(config.train.log_dir, 'pretext')
    pretext_model_path = os.path.join(pretext_dir, 'last.ckpt')

    topk_neighbors_train_path = os.path.join(
        pretext_dir, 'topk_neighbors_train.npy')
    topk_neighbors_val_path = os.path.join(
        pretext_dir, 'topk_neighbors_val.npy')
    evaluate_intervals = config.train.evaluate

    log_dir = os.path.join(config.train.log_dir, 'neighbors')
    os.makedirs(log_dir, exist_ok=True)

    # For reproducibility
    pl.seed_everything(config.seed)
    use_wandb = config.wandb

    train_transformations = get_train_transformations(config, task='cutout')
    val_transformations = get_val_transformations(config)
    train_dataset = get_train_dataset(
        config, train_transformations, neighbors=True, topk_neighbors_train_path=topk_neighbors_train_path)
    val_dataset = get_val_dataset(
        config, val_transformations, neighbors=True, topk_neighbors_val_path=topk_neighbors_val_path)
    train_dataloader = DataLoader(train_dataset,
                                  config.train.batch_size,
                                  num_workers=config.train.num_workers,
                                  shuffle=True,
                                  pin_memory=True,
                                  drop_last=True)
    val_dataloader = DataLoader(val_dataset,
                                config.train.batch_size,
                                num_workers=config.train.num_workers,
                                shuffle=False,
                                drop_last=False,
                                pin_memory=True)

    # Create contrastive model.
    model = ConsistencyEncoder(config)
    model.load_pretext_encoder(torch.load(pretext_model_path, map_location='cpu')['state_dict'])
    
    if use_wandb:
        wandb_logger = WandbLogger(
            project=config.project_name, name=f"{config.experiment_name}-{config.note}", config=config)
        wandb_logger.watch(model)

    checkpoint_callback = ModelCheckpoint(monitor='train_loss',
                                          filename='{epoch}-{train_loss:.2f}',
                                          dirpath=log_dir,
                                          save_last=True,
                                          save_weights_only=False)

    trainer = pl.Trainer(accelerator='gpu',
                         strategy='ddp',
                         devices=config.train.devices,
                         precision='16-mixed',
                         max_epochs=config.train.epochs,
                         enable_progress_bar=config.verbose,
                         check_val_every_n_epoch=evaluate_intervals,
                         deterministic=True,
                         use_distributed_sampler=True,
                         logger=wandb_logger if use_wandb else None,
                         default_root_dir=log_dir,
                         enable_checkpointing=True,
                         callbacks=[RichProgressBar(), checkpoint_callback]
                         )
    # Checkpoint
    if config.train.resume and config.train.ckpt_path:
        trainer.fit(model, train_dataloaders=train_dataloader,
                    val_dataloaders=val_dataloader, ckpt_path=config.train.ckpt_path)
    else:
        trainer.fit(model, train_dataloaders=train_dataloader,
                    val_dataloaders=val_dataloader)

    del trainer

    if LOCAL_RANK == 0 or LOCAL_RANK == -1:
        # Mine neighbors information and save them.
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
        model = ConsistencyEncoder.load_from_checkpoint(
            os.path.join(log_dir, 'last.ckpt'), map_location=device, args=config)

        # Evaluate and save the final model
        print('Evaluate best model based on SCAN metric at the end')

        predictions = get_predictions(
            config, val_dataloader, model, have_neighbors=True)
        neighbors_stats = neighbors_evaluate(predictions)
        lowest_loss_head = neighbors_stats['lowest_loss_head']
        print(neighbors_stats)

        clustering_stats = hungarian_evaluate(lowest_loss_head, predictions,
                                              class_names=val_dataset.dataset.classes,
                                              compute_confusion_matrix=True,
                                              confusion_matrix_file=os.path.join(log_dir, 'confusion_matrix.png'))
        print(clustering_stats)


if __name__ == '__main__':
    main()

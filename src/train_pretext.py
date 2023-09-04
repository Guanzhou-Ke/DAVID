import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar
import lightning.pytorch as pl

from configs.basic_config import get_cfg
from models.contrastive_models import ContrastiveModel
from utils.datatool import (get_train_transformations,
                            get_val_transformations,
                            get_train_dataset,
                            get_val_dataset)

from utils.memory_bank import MemoryBank
from utils.misc import fill_memory_bank

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

    log_dir = os.path.join(config.train.log_dir, 'pretext')
    os.makedirs(log_dir, exist_ok=True)

    use_wandb = config.wandb
    evaluate_intervals = config.train.evaluate
    # For reproducibility
    pl.seed_everything(config.seed, workers=True)

    # Create Dataset
    train_transforms = get_train_transformations(config, task='pretext')
    val_transforms = get_val_transformations(config)
    train_dataset = get_train_dataset(
        config, train_transforms, neighbors=False)
    val_dataset = get_val_dataset(config, val_transforms)
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
    model = ContrastiveModel(config, mode='train')
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
                         check_val_every_n_epoch=evaluate_intervals,
                         enable_progress_bar=config.verbose,
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
        device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device('cpu')
        model = ContrastiveModel.load_from_checkpoint(
            os.path.join(log_dir, 'last.ckpt'), map_location=device)
        topk_neighbors_train_path = os.path.join(
            log_dir, 'topk_neighbors_train.npy')
        topk_neighbors_val_path = os.path.join(
            log_dir, 'topk_neighbors_val.npy')

        # Memory Bank
        # Dataset w/o augs for knn eval
        base_dataset = get_train_dataset(config, val_transforms)
        base_dataloader = DataLoader(base_dataset,
                                     config.train.batch_size,
                                     num_workers=config.train.num_workers,
                                     shuffle=False,
                                     drop_last=False,
                                     pin_memory=True)
        memory_bank_base = MemoryBank(len(base_dataset), config, device)
        memory_bank_base.to(device)
        memory_bank_val = MemoryBank(len(val_dataset), config, device)
        memory_bank_val.to(device)

        print('Fill memory bank for mining the nearest neighbors (train) ...')
        fill_memory_bank(base_dataloader, model, memory_bank_base, device)
        topk = 20
        print('Mine the nearest neighbors (Top-%d)' % (topk))
        indices, acc = memory_bank_base.mine_nearest_neighbors(topk)
        print('Accuracy of top-%d nearest neighbors on train set is %.2f' %
              (topk, 100*acc))
        np.save(topk_neighbors_train_path, indices)

        # Mine the topk nearest neighbors at the very end (Val)
        # These will be used for validation.
        print('Fill memory bank for mining the nearest neighbors (val) ...')
        fill_memory_bank(val_dataloader, model, memory_bank_val, device)
        topk = 5
        print('Mine the nearest neighbors (Top-%d)' % (topk))
        indices, acc = memory_bank_val.mine_nearest_neighbors(topk)
        print('Accuracy of top-%d nearest neighbors on val set is %.2f' %
              (topk, 100*acc))
        np.save(topk_neighbors_val_path, indices)


if __name__ == "__main__":
    main()

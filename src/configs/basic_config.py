
import os

from yacs.config import CfgNode as CN


# Basic config, including basic information, dataset, and training setting.
_C = CN()

# project name, for wandb's records. **Required**
_C.project_name = 'DAVID'
# project description, what problem does this project tackle?
_C.project_desc = '' 
# seed
_C.seed = 42
# print log
_C.verbose = True
# enable wandb:
_C.wandb = True
# experiment name.
_C.experiment_name = "None"
# For multi-view setting
_C.views = 2
# Network output dim.
_C.hidden_dim = 128
# Experiment Notes.
_C.note = ""


# Network setting.
_C.backbone = CN()
_C.backbone.type = "cnn"
# normalizations ['batch', 'layer']
_C.backbone.normalization = 'batch'
# activation
_C.backbone.activation = 'relu'
# using maxpooling in resnet.
_C.backbone.max_pooling = False
# Network initialize method, 'xavier' | 'gaussian' | 'kaiming' | 'orthogonal'
# default 'kaiming'.
_C.backbone.init_method = 'kaiming'
# enbale shared view-specific encoder
_C.backbone.fc_identity = True
# for cnn.
_C.backbone.in_channel = 1


# For dataset
_C.dataset = CN()
# ['Scene-15', 'LandUse-21', 'Caltech101-20', 'NoisyMNIST', 
# 'EdgeMnist', 'FashionMnist', 'coil-20', 'coil-100', 'DHA23', "UWA30"]
_C.dataset.name = 'EdgeMnist'
# rootdir
_C.dataset.root = './data'
# class_num
_C.dataset.class_num = 10



# For augmentation
# training augmentation
_C.training_augmentation = CN()
_C.training_augmentation.crop_size = 32
# Need training augmentation? such as mnist we suggest set as false.
_C.training_augmentation.hflip = True
# random resize crop:
_C.training_augmentation.random_resized_crop = CN()
_C.training_augmentation.random_resized_crop.size = 32
_C.training_augmentation.random_resized_crop.scale = [0.2, 1.0]
# color jitter random apply:
_C.training_augmentation.color_jitter_random_apply = CN()
_C.training_augmentation.color_jitter_random_apply.p = 0.8
# color jitter
_C.training_augmentation.color_jitter = CN()
_C.training_augmentation.color_jitter.brightness = 0.4
_C.training_augmentation.color_jitter.contrast = 0.4
_C.training_augmentation.color_jitter.saturation = 0.4
_C.training_augmentation.color_jitter.hue = 0.1
# random_grayscale
_C.training_augmentation.random_grayscale = CN()
_C.training_augmentation.random_grayscale.p = 0.2
# RandAugment config.
_C.training_augmentation.num_strong_augs = 4
# Cutout config
_C.training_augmentation.cutout_kwargs = CN()
_C.training_augmentation.cutout_kwargs.n_holes = 1
_C.training_augmentation.cutout_kwargs.length = 16
_C.training_augmentation.cutout_kwargs.random = True
# normalize
_C.training_augmentation.normalize = CN()
_C.training_augmentation.normalize.mean = [0.4914, 0.4822, 0.4465]
_C.training_augmentation.normalize.std = [0.2023, 0.1994, 0.2010]

# validation augmentation
_C.valid_augmentation = CN()
# center crop size
_C.valid_augmentation.crop_size = 32
_C.valid_augmentation.use_normalize = True
# normalize
_C.valid_augmentation.normalize = CN()
_C.valid_augmentation.normalize.mean = [0.4914, 0.4822, 0.4465]
_C.valid_augmentation.normalize.std = [0.2023, 0.1994, 0.2010]


# for training.
_C.train = CN()
_C.train.epochs = 100
_C.train.batch_size = 512
_C.train.optim = 'sgd'
_C.train.devices = [0, 1]
_C.train.lr = 0.001
_C.train.num_workers = 4
_C.train.save_log = True
# if None, it will be set as './experiments/results/[model name]/[dataset name]'
_C.train.log_dir = ""
# For test embedding dataset
_C.train.save_embeddings = -1
# for incomplete setting, float. [0-1], 0 denotes complete.
_C.train.missing_ratio = 0.
# the interval of evaluate epoch, defaults to 5.
_C.train.evaluate = 5
# Learning rate scheduler, [cosine, step]
_C.train.scheduler = 'cosine'
_C.train.lr_decay_rate = 0.1
_C.train.lr_decay_epochs = 30
# for knn neighbors.
_C.train.knn_num = 200
_C.train.knn_temperature = .5
# samling num.
_C.train.samples_num = 6
# using checkpoint training.
_C.train.resume = False
_C.train.ckpt_path = ""


# commom feature pooling method. mean, sum, or first
_C.fusion = CN()
# C -> only consistency, V -> only view-specific, CV -> [C x V]
_C.fusion.type = 'C'
_C.fusion.pooling_method = 'sum'
# view specific fusion weight
_C.fusion.vs_weights = 1.


# for clustering setting.
_C.clustering = CN()
_C.clustering.nheads = 1
_C.clustering.entropy_weight = 2.
_C.clustering.num_neighbors = 5

# for consistency encoder setting.
_C.consis_enc = CN()
_C.consis_enc.enable = True
_C.consis_enc.backbone = 'resnet18'
_C.consis_enc.in_channel = 1
_C.consis_enc.max_pooling = False
_C.consis_enc.activation = 'relu'
_C.consis_enc.first_norm = False
_C.consis_enc.output_dim = 512
_C.consis_enc.project_dim = 128
# 'dcl' or 'dclw'
_C.consis_enc.loss_type = 'dcl'
_C.consis_enc.temperature = 0.1
# stop training epoch.
_C.consis_enc.freeze_epoch = 90


# for view-specific encoder setting.
_C.vspecific = CN()
_C.vspecific.enable = True
# list or str, str denotes all encoder type are same.
_C.vspecific.hidden_dims = [32, 64, 128, 256, 512]
# latent dim expands.
_C.vspecific.expands = 4
_C.vspecific.latent_dim = 128
_C.vspecific.kld_weight = 0.00025
# best view for concatenating consistency representation and view-specific representation.
_C.vspecific.best_view = 0

# disentanglement
_C.disent = CN()
_C.disent.lam = 1.
_C.disent.x_dim = 10
_C.disent.y_dim = 10
_C.disent.hidden_size = 100
_C.disent.alpha = 0.01
_C.disent.mode = 'bias'



def get_cfg(config_file_path):
    """
    Initialize configuration.
    """
    config = _C.clone()
    # merge specific config.
    config.merge_from_file(config_file_path)
    
    if not config.train.log_dir:
        path = f'./experiments/{config.experiment_name}'
        os.makedirs(path, exist_ok=True)
        config.train.log_dir = path
    else:
        os.makedirs(config.train.log_dir, exist_ok=True)
    config.freeze()
    return config
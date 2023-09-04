import lightning.pytorch as pl
from lightning.pytorch import Trainer, LightningModule


class MemCallback(pl.Callback):

    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        pl_module.mem_ptr[0] = 0
        device = pl_module.device
        for features, y in pl_module.mem_list:
            features = features.to(device)
            y = y.to(device)
            pl_module._update_mem(features, y)
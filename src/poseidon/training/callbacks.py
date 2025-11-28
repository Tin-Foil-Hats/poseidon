import lightning.pytorch as L

class SetEpochOnIterable(L.Callback):
    """Propagates trainer.current_epoch to a train IterableDataset that implements set_epoch(int)."""
    def on_train_epoch_start(self, trainer, pl_module):
        dm = trainer.datamodule
        ds = getattr(dm, "train_ds", None)
        if hasattr(ds, "set_epoch"):
            ds.set_epoch(trainer.current_epoch)

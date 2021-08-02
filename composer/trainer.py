from pathlib import Path

import pandas as pd

import pytorch_lightning as pl


class CustomTrainer(pl.Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def export_csv_log(
        self, columns=[
            'epoch', 'train_loss', 'train_mse_loss', 'train_kl_loss',
            'val_loss', 'val_mse_loss', 'val_kl_loss', 'lr', 'kl_lambda',
            'kl_ramp_strength'
        ]
    ):
        """Custom method for exporting the trainer logs to something much more
        readable. Only executes on the 0th global rank for DDP jobs."""

        if self.global_rank > 0:
            return

        metrics = self.logger.experiment.metrics
        log_dir = self.logger.experiment.log_dir

        path = Path(log_dir) / Path("custom_metrics.csv")
        t = pd.DataFrame([d for d in metrics if 'train_loss' in d])
        v = pd.DataFrame([d for d in metrics if 'val_loss' in d])
        df = pd.concat([t, v], join='outer', axis=1)
        df = df.loc[:, ~df.columns.duplicated()]
        df = df[columns]
        df.to_csv(path, index=False)

    def get_best_model_checkpoint_path(self):
        """Returns the checkpoint path corresponding to the best model.

        Returns
        -------
        str
        """

        path = Path(self.logger.experiment.log_dir) / Path("checkpoints")
        checkpoints = list(path.iterdir())
        checkpoints.sort()
        return str(checkpoints[-1])

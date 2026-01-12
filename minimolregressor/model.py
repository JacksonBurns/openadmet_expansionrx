import numpy as np
import pandas as pd
import torch
from astartes import train_test_split
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.pipeline import Pipeline
from torch.utils.data import DataLoader, TensorDataset


class SmilesToEmbeddingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, embedding_parquet: str):
        self.embedding_parquet = embedding_parquet
        self.embedding_df = pd.read_parquet(embedding_parquet)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        try:
            emb = self.embedding_df.loc[X].to_numpy(dtype=np.float32)
        except KeyError as e:
            missing = list(set(X) - set(self.embedding_df.index))
            raise ValueError(
                f"Missing embeddings for SMILES in file {self.embedding_parquet}: {missing}"
            ) from e
        return torch.from_numpy(emb)


class minimolTaskHead(LightningModule):
    def __init__(self, learning_rate: float = 0.0003, n_tasks: int = 1):
        super().__init__()
        self.learning_rate = learning_rate
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(512, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.10),
            torch.nn.Linear(512, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.10),
        )
        _readout_modules = [torch.nn.Linear(1024, n_tasks)]
        self.readout = torch.nn.Sequential(*_readout_modules)
        self.save_hyperparameters()

    def configure_optimizers(self):
        return {"optimizer": torch.optim.Adam(self.parameters(), lr=self.learning_rate)}

    def forward(self, x):
        return self.readout(torch.cat((x, self.mlp(x)), dim=1))

    def _step(self, batch, name):
        x, y = batch
        y_hat = self(x)
        
        mask = ~torch.isnan(y)
        if mask.sum() == 0:
            loss = (y_hat * 0.0).sum()
        else:
            loss = torch.nn.functional.mse_loss(y_hat[mask], y[mask])
        
        self.log(f"{name}/loss", loss, prog_bar=True)
        return loss


    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "validation")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")

    def predict_step(self, batch, batch_idx):
        return self(batch[0])


class MinimolLightningEstimator(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        batch_size=64,
        max_epochs=50,
        learning_rate=3e-4,
        random_seed=42,
        output_dir="minimol_output",
        n_tasks=1,
    ):
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.random_seed = random_seed
        self.output_dir = output_dir
        self.n_tasks = n_tasks

        # for stately loading and inference
        self.model_ = None

    def fit(self, X, y):
        y = torch.tensor(y, dtype=torch.float32)
        train_idx, val_idx = train_test_split(
            np.arange(len(X)), train_size=0.8, random_state=self.random_seed
        )
        train_ds = TensorDataset(X[train_idx], y[train_idx])
        val_ds = TensorDataset(X[val_idx], y[val_idx])

        train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, drop_last=True)
        val_dl = DataLoader(val_ds, batch_size=self.batch_size)

        self.model_ = minimolTaskHead(
            learning_rate=self.learning_rate, n_tasks=self.n_tasks
        )

        callbacks = [
            EarlyStopping(monitor="validation/loss", mode="min", patience=5),
            ModelCheckpoint(dirpath=self.output_dir, monitor="validation/loss", mode="min", save_top_k=1),
        ]

        logger = TensorBoardLogger(self.output_dir, name="minimol", default_hp_metric=False)

        self.trainer_ = Trainer(
            max_epochs=self.max_epochs, callbacks=callbacks, logger=logger, log_every_n_steps=1
        )

        self.trainer_.fit(self.model_, train_dl, val_dl)
        self.best_ckpt_ = self.trainer_.checkpoint_callback.best_model_path

        self.model_ = minimolTaskHead.load_from_checkpoint(self.best_ckpt_)
        self.model_.eval()

        return self

    @torch.no_grad()
    def predict(self, X):

        dl = DataLoader(X, batch_size=self.batch_size)
        preds = []

        for batch in dl:
            batch = batch.to(self.model_.device)
            preds.append(self.model_(batch).cpu())

        return torch.cat(preds).numpy()

    def _more_tags(self):
        return {"multioutput": True, "requires_y": True}


def get_minimol_pipe(embedding_parquet: str, output_dir: str, random_seed: int = 42, n_tasks: int = 1):
    return Pipeline(
        [
            ("smiles2emb", SmilesToEmbeddingTransformer(embedding_parquet)),
            ("minimol", MinimolLightningEstimator(output_dir=output_dir, random_seed=random_seed, n_tasks=n_tasks, max_epochs=100)),
        ]
    )

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from astartes import train_test_split
from lightning import LightningModule, Trainer, seed_everything
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
    def __init__(self, learning_rate: float = 0.0003, n_tasks: int = 1, weights=None):
        super().__init__()
        self.learning_rate = learning_rate
        self.register_buffer("weights", torch.tensor(weights) if weights is not None else None)
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
            loss = torch.tensor(0.0, device=self.device)
        else:
            if self.weights is not None:
                y *= self.weights
                y_hat *= self.weights
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


class MinimolCrossValLightningEstimator(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        batch_size=64,
        max_epochs=50,
        learning_rate=3e-4,
        random_seed=42,
        output_dir="minimol_output",
        n_tasks=1,
        n_ensemble=3,
        weights=None,
    ):
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.random_seed = random_seed
        self.output_dir = output_dir
        self.n_tasks = n_tasks
        self.n_ensemble = n_ensemble
        self.weights = weights

        # fitted state
        self.models_ = []
        self.best_ckpts_ = []

    def fit(self, X, y):
        y = torch.tensor(y, dtype=torch.float32)

        self.models_ = []
        self.best_ckpts_ = []

        for k in range(self.n_ensemble):
            seed = self.random_seed + k
            seed_everything(seed)

            train_idx, val_idx = train_test_split(
                np.arange(len(X)), train_size=0.9, random_state=seed
            )

            train_ds = TensorDataset(X[train_idx], y[train_idx])
            val_ds = TensorDataset(X[val_idx], y[val_idx])

            train_dl = DataLoader(
                train_ds, batch_size=self.batch_size, shuffle=True, drop_last=True
            )
            val_dl = DataLoader(val_ds, batch_size=self.batch_size)

            model = minimolTaskHead(
                learning_rate=self.learning_rate, n_tasks=self.n_tasks, weights=self.weights
            )

            run_dir = Path(self.output_dir) / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)

            callbacks = [
                EarlyStopping(monitor="validation/loss", mode="min", patience=5),
                ModelCheckpoint(
                    dirpath=run_dir, monitor="validation/loss", mode="min", save_top_k=1
                ),
            ]

            logger = TensorBoardLogger(save_dir=run_dir, name="minimol", default_hp_metric=False)

            trainer = Trainer(
                max_epochs=self.max_epochs,
                callbacks=callbacks,
                logger=logger,
                log_every_n_steps=1,
                deterministic=True,
            )

            trainer.fit(model, train_dl, val_dl)

            best_ckpt = trainer.checkpoint_callback.best_model_path
            model = minimolTaskHead.load_from_checkpoint(best_ckpt)
            model.eval()

            self.models_.append(model)
            self.best_ckpts_.append(best_ckpt)

        return self

    @torch.no_grad()
    def predict(self, X):
        if not self.models_:
            raise RuntimeError("Estimator has not been fitted.")

        dl = DataLoader(X, batch_size=self.batch_size)
        all_preds = []

        for model in self.models_:
            preds = []
            for batch in dl:
                batch = batch.to(model.device)
                preds.append(model(batch).cpu())
            all_preds.append(torch.cat(preds))

        # shape: (n_models, n_samples, n_tasks)
        stacked = torch.stack(all_preds, dim=0)

        # ensemble mean
        return stacked.mean(dim=0).numpy()

    def _more_tags(self):
        return {"multioutput": True, "requires_y": True}


def get_minimol_pipe(
    embedding_parquet: str, output_dir: str, random_seed: int = 42, n_tasks: int = 1, weights=None
):
    return Pipeline(
        [
            ("smiles2emb", SmilesToEmbeddingTransformer(embedding_parquet)),
            (
                "minimol",
                MinimolCrossValLightningEstimator(
                    output_dir=Path(output_dir) / "minimol",
                    random_seed=random_seed,
                    n_tasks=n_tasks,
                    max_epochs=100,
                    weights=weights,
                ),
            ),
        ]
    )

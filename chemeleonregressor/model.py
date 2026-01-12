import os
from argparse import ArgumentParser, Namespace
from datetime import datetime
from os import PathLike
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from astartes import train_test_split
from chemprop.cli.common import add_common_args
from chemprop.cli.train import add_train_args, build_model
from chemprop.cli.utils.parsing import make_dataset
from chemprop.data.collate import collate_batch
from chemprop.data.datapoints import MoleculeDatapoint
from chemprop.nn.transforms import UnscaleTransform
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from rdkit import Chem
from scikit_mol.conversions import SmilesToMolTransformer
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline
from torch.utils.data import DataLoader

NOW = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
CHEMPROP_TRAIN_DIR = Path(
    os.getenv("CHEMPROP_TRAIN_DIR", Path(__file__).parent / "chemprop_training")
)


def add_train_defaults(args: Namespace) -> Namespace:
    parser = ArgumentParser()
    parser = add_common_args(parser)
    parser = add_train_args(parser)
    defaults = parser.parse_args([])
    for k, v in vars(defaults).items():
        if not hasattr(args, k):
            setattr(args, k, v)
    return args


class ChemeleonRegressor(RegressorMixin, BaseEstimator):
    def __init__(
        self,
        num_workers: int = 0,
        batch_size: int = 64,
        output_dir: Optional[PathLike] = CHEMPROP_TRAIN_DIR / "sklearn_output" / NOW,
        ffn_hidden_dim: int = 2_048,
        ffn_num_layers: int = 1,
        accelerator: str = "auto",
        devices: str | int | Sequence[int] = "auto",
        epochs: int = 50,
        random_seed: int = 42,
        n_tasks: int = 1,
    ):
        args = Namespace(
            num_workers=num_workers,
            batch_size=batch_size,
            output_dir=output_dir,
            ffn_hidden_dim=ffn_hidden_dim,
            ffn_num_layers=ffn_num_layers,
            accelerator=accelerator,
            devices=devices,
            epochs=epochs,
            from_foundation="chemeleon",
            num_tasks=n_tasks,
        )
        self.output_dir = output_dir
        self.random_seed = random_seed
        self.args = add_train_defaults(args)
        self.model = None
        for name, value in locals().items():
            if name not in {"self", "args"}:
                setattr(self, name, value)

    def _build_dps(self, X: np.ndarray[Chem.Mol], Y: np.ndarray | None):
        if Y is None:
            return [MoleculeDatapoint(mol=mol) for mol in X.flatten()]
        return [MoleculeDatapoint(mol=mol, y=target) for mol, target in zip(X.flatten(), Y)]

    def __sklearn_is_fitted__(self):
        return True

    def transform(self, X):
        return self.predict(X)

    def fit(self, X, y):
        train_idx, val_idx = train_test_split(
            np.arange(len(X)), train_size=0.8, random_state=self.random_seed
        )
        train_datapoints = self._build_dps(X[train_idx], y[train_idx])
        val_datapoints = self._build_dps(X[val_idx], y[val_idx])
        train_set = make_dataset(train_datapoints)
        val_set = make_dataset(val_datapoints)
        if self.model is None:
            output_scaler = train_set.normalize_targets()
            val_set.normalize_targets(output_scaler)
            output_transform = UnscaleTransform.from_standard_scaler(output_scaler)
            self.model = build_model(self.args, train_set, output_transform, [None] * 4)
        train_loader = DataLoader(
            train_set,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            collate_fn=collate_batch,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            collate_fn=collate_batch,
        )
        callbacks = [
            EarlyStopping(monitor="val_loss", mode="min", patience=5),
            ModelCheckpoint(dirpath=self.output_dir, monitor="val_loss", mode="min", save_top_k=1),
        ]
        trainer = Trainer(
            accelerator=self.args.accelerator,
            devices=self.args.devices,
            max_epochs=self.args.epochs,
            callbacks=callbacks,
            logger=TensorBoardLogger(save_dir=self.output_dir, name="logs", default_hp_metric=False),
        )
        trainer.fit(self.model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        # reload best model
        self.model = self.model.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
        self.model.eval()
        return self

    def predict(self, X):
        datapoints = self._build_dps(X, None)
        test_set = make_dataset(datapoints)
        self._y = test_set.Y
        dl = DataLoader(
            test_set,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            collate_fn=collate_batch,
        )
        eval_trainer = Trainer(
            accelerator=self.args.accelerator, devices=1, enable_progress_bar=True, logger=False
        )
        preds = eval_trainer.predict(self.model, dataloaders=dl, return_predictions=True)
        return torch.cat(preds, dim=0).numpy(force=True)

    def _more_tags(self):
        return {"multioutput": True, "requires_y": True}


def get_chemeleon_pipe(outdir: str, random_seed: int = 42, n_tasks: int = 1):
    return Pipeline(
        [
            ("smiles2mol", SmilesToMolTransformer(n_jobs=-1)),
            ("chemeleon", ChemeleonRegressor(random_seed=random_seed, n_tasks=n_tasks, output_dir=outdir)),
        ]
    )

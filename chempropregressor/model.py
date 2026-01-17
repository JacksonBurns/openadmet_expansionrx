import os
from argparse import ArgumentParser, Namespace
from datetime import datetime
from os import PathLike, name
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from astartes import train_test_split
from chemprop.cli.common import add_common_args
from chemprop.cli.train import add_train_args, build_model, normalize_inputs
from chemprop.cli.utils.parsing import make_dataset
from chemprop.data.collate import collate_batch
from chemprop.data.datapoints import MoleculeDatapoint
from chemprop.nn.transforms import UnscaleTransform
from lightning.pytorch import Trainer
from lightning import seed_everything
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


class ChempropCrossValRegressor(RegressorMixin, BaseEstimator):
    def __init__(
        self,
        num_workers: int = 0,
        batch_size: int = 64,
        output_dir: Optional[PathLike] = CHEMPROP_TRAIN_DIR / "sklearn_output" / NOW,
        accelerator: str = "auto",
        devices: str | int | Sequence[int] = "auto",
        epochs: int = 50,
        random_seed: int = 42,
        n_tasks: int = 1,
        n_ensemble: int = 5,
        weights: Optional[Sequence[float]] = None,
        config: Optional[dict] = None,
    ):
        args = Namespace(
            num_workers=num_workers,
            batch_size=batch_size,
            output_dir=output_dir,
            accelerator=accelerator,
            devices=devices,
            epochs=epochs,
            num_tasks=n_tasks,
        )
        if weights is not None:
            args.weights = weights
        if config is None:
            args.from_foundation = "chemeleon"
        else:
            for k, v in config.items():
                setattr(args, k, v)

        self.output_dir = output_dir
        self.random_seed = random_seed
        self.n_ensemble = n_ensemble
        self.args = add_train_defaults(args)

        self.models_: list = []
        self.best_ckpts_: list = []

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
        self.models_ = []
        self.best_ckpts_ = []

        for k in range(self.n_ensemble):
            seed = self.random_seed + k
            seed_everything(seed)

            train_idx, val_idx = train_test_split(
                np.arange(len(X)), train_size=0.9, random_state=seed
            )

            train_dps = self._build_dps(X[train_idx], y[train_idx])
            val_dps = self._build_dps(X[val_idx], y[val_idx])

            train_set = make_dataset(train_dps)
            val_set = make_dataset(val_dps)

            output_scaler = train_set.normalize_targets()
            val_set.normalize_targets(output_scaler)

            output_transform = UnscaleTransform.from_standard_scaler(output_scaler)

            input_transforms = normalize_inputs(train_set, val_set, self.args)

            model = build_model(self.args, train_set, output_transform, input_transforms)

            run_dir = Path(self.output_dir) / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)

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
                ModelCheckpoint(dirpath=run_dir, monitor="val_loss", mode="min", save_top_k=1),
            ]

            trainer = Trainer(
                accelerator=self.args.accelerator,
                devices=self.args.devices,
                max_epochs=self.args.epochs,
                callbacks=callbacks,
                logger=TensorBoardLogger(save_dir=run_dir, name="logs", default_hp_metric=False),
                deterministic=True,
            )

            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

            best_ckpt = trainer.checkpoint_callback.best_model_path
            model = model.load_from_checkpoint(best_ckpt)
            model.eval()

            self.models_.append(model)
            self.best_ckpts_.append(best_ckpt)

        return self

    def predict(self, X):
        if not self.models_:
            raise RuntimeError("Estimator has not been fitted.")

        datapoints = self._build_dps(X, None)
        test_set = make_dataset(datapoints)

        dl = DataLoader(
            test_set,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            collate_fn=collate_batch,
        )

        all_preds = []

        for model in self.models_:
            eval_trainer = Trainer(
                accelerator=self.args.accelerator,
                devices=1,
                enable_progress_bar=False,
                logger=False,
                deterministic=True,
            )
            preds = eval_trainer.predict(model, dataloaders=dl, return_predictions=True)
            all_preds.append(torch.cat(preds, dim=0))

        # (n_models, n_samples, n_tasks)
        stacked = torch.stack(all_preds, dim=0)

        return stacked.mean(dim=0).numpy(force=True)

    def _more_tags(self):
        return {"multioutput": True, "requires_y": True}


def get_chemprop_pipe(
    config: str | dict, outdir: str, random_seed: int = 42, n_tasks: int = 1, weights=None
):
    if config == "chemeleon":
        return Pipeline(
            [
                ("smiles2mol", SmilesToMolTransformer(n_jobs=-1)),
                (
                    "chemeleon",
                    ChempropCrossValRegressor(
                        random_seed=random_seed,
                        n_tasks=n_tasks,
                        output_dir=Path(outdir) / "chemeleon",
                        weights=weights,
                        config=None,  # uses chemeleon defaults
                    ),
                ),
            ]
        )
    else:
        return Pipeline(
            [
                ("smiles2mol", SmilesToMolTransformer(n_jobs=-1)),
                (
                    "chemprop",
                    ChempropCrossValRegressor(
                        random_seed=random_seed,
                        n_tasks=n_tasks,
                        output_dir=Path(outdir) / "chemprop",
                        weights=weights,
                        config=config,
                    ),
                ),
            ]
        )

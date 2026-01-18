import sys
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd

from chempropregressor.model import get_chemprop_pipe
from minimolregressor.model import get_minimol_pipe
from physicoforestregressor.model import get_prf_pipe
from multitask_stacking_regressor import MultitaskStackingRegressor, CentralScrutinizer

TASKS = [
    "LogD",
    "KSOL",
    "HLM CLint",
    "MLM CLint",
    "Caco-2 Permeability Papp A>B",
    "Caco-2 Permeability Efflux",
    "MPPB",
    "MBPB",
    "MGMB",
]

if __name__ == "__main__":
    try:
        outdir = Path(sys.argv[1])
        outdir.mkdir(parents=True, exist_ok=True)
    except:
        print("Usage: python train.py <output_directory>")
        exit(1)

    outdir /= datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    outdir.mkdir()

    df = pd.read_csv("train.csv")

    estimators = [
        (
            "chemprop",
            get_chemprop_pipe(
                config={
                    "activation": "PRELU",
                    "ffn_hidden_dim": 800,
                    "batch_size": 64,
                    "max_lr": 0.0008,
                    "ffn_num_layers": 2,
                    "warmup_epochs": 16,
                    "final_lr": 0.0001,
                    "aggregation": "sum",
                    "init_lr": 0.0003,
                    "depth": 6,
                    "message_hidden_dim": 1300,
                },
                outdir=outdir,
                n_tasks=len(TASKS),
            ),
        ),
        ("chemeleon", get_chemprop_pipe(config="chemeleon", outdir=outdir, n_tasks=len(TASKS))),
        (
            "minimol",
            get_minimol_pipe("minimol_features.parquet", output_dir=outdir, n_tasks=len(TASKS)),
        ),
        ("prf", get_prf_pipe()),
    ]

    model = MultitaskStackingRegressor(
        estimators=estimators,
        final_estimator=CentralScrutinizer(random_state=42, output_dir=outdir, max_epochs=1024),
        n_folds=5,
        shuffle=True,
        random_state=42,
    )
    model.fit(df["clean_smiles"], df[TASKS].to_numpy())
    outmodel = outdir / "model.joblib"
    joblib.dump(model, outmodel)

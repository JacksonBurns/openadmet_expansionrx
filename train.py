import sys
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd

from chemeleonregressor.model import get_chemeleon_pipe
from minimolregressor.model import get_minimol_pipe
from physicoforestregressor.model import get_prf_pipe, ColumnwiseRFRegressor
from multitask_stacking_regressor import MultitaskStackingRegressor

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
        ("prf", get_prf_pipe()),
        ("minimol", get_minimol_pipe("minimol_features.parquet", output_dir=outdir, n_tasks=len(TASKS))),
        ("chemeleon", get_chemeleon_pipe(outdir=outdir, n_tasks=len(TASKS))),
    ]

    model = MultitaskStackingRegressor(
        estimators=estimators,
        final_estimator=ColumnwiseRFRegressor(n_jobs=-1, random_state=42),
        n_folds=5,
        shuffle=True,
        random_state=42,
    )
    model.fit(
        df["clean_smiles"],
        df[TASKS].to_numpy(),
    )
    outmodel = outdir / "model.joblib"
    joblib.dump(model, outmodel)

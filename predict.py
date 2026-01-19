import sys
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

if __name__ == "__main__":
    try:
        model_0_path = sys.argv[1]
        model_1_path = sys.argv[2]
    except:
        print("Usage: python predict.py <path_to_model_0> <path_to_model_1>")
        exit(1)
    model_0 = joblib.load(model_0_path)
    model_1 = joblib.load(model_1_path)
    test_df = pd.read_csv("test.csv")
    predictions_0 = model_0.predict(test_df["clean_smiles"])
    df_0 = pd.DataFrame(
        data=predictions_0,
        columns=[
            "LogD",
            "KSOL",
        ],
    )
    predictions_1 = model_1.predict(test_df["clean_smiles"])
    df_1 = pd.DataFrame(
        data=predictions_1,
        columns=[
            "HLM CLint",
            "MLM CLint",
            "Caco-2 Permeability Papp A>B",
            "Caco-2 Permeability Efflux",
            "MPPB",
            "MBPB",
            "MGMB",
        ],
    )
    df = pd.concat([df_0, df_1], axis=1)
    # clip predictions to the bounds observed during training (clipped, by me)
    train = pd.read_csv("train.csv")
    train_bounds = train[
        [
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
    ].agg(["min", "max"])
    for col in [
        "LogD",
        "KSOL",
        "HLM CLint",
        "MLM CLint",
        "Caco-2 Permeability Papp A>B",
        "Caco-2 Permeability Efflux",
        "MPPB",
        "MBPB",
        "MGMB",
    ]:
        df[col] = df[col].clip(
            lower=train_bounds.loc["min", col], upper=train_bounds.loc["max", col]
        )

    # undo log transforms
    for col in [
        "KSOL",
        "HLM CLint",
        "MLM CLint",
        "Caco-2 Permeability Papp A>B",
        "Caco-2 Permeability Efflux",
        "MPPB",
        "MBPB",
        "MGMB",
    ]:
        df[col] = np.expm1(df[col])
    df["Molecule Name"] = test_df["Molecule Name"]
    df["SMILES"] = test_df["SMILES"]
    df[
        [
            "Molecule Name",
            "SMILES",
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
    ].to_csv(f"test_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", index=False)

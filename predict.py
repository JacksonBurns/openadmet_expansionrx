import sys
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

if __name__ == "__main__":
    try:
        model_path = sys.argv[1]
    except:
        print("Usage: python predict.py <path_to_model>")
        exit(1)
    model = joblib.load(model_path)
    test_df = pd.read_csv("test.csv")
    predictions = model.predict(test_df["clean_smiles"])
    df = pd.DataFrame(
        data=predictions,
        columns=[
            "LogD",
            "KSOL",
            "HLM CLint",
            "MLM CLint",
            "Caco-2 Permeability Papp A>B",
            "Caco-2 Permeability Efflux",
            "MPPB",
            "MBPB",
            "MGMB",
        ],
    )
    # clip predictions to the bounds observed during training (clipped, by me)
    train = pd.read_csv("train.csv")
    train_bounds = train[["LogD","KSOL","HLM CLint","MLM CLint","Caco-2 Permeability Papp A>B","Caco-2 Permeability Efflux","MPPB","MBPB","MGMB"]].agg(["min","max"])
    for col in ["LogD","KSOL","HLM CLint","MLM CLint","Caco-2 Permeability Papp A>B","Caco-2 Permeability Efflux","MPPB","MBPB","MGMB"]:
        df[col] = df[col].clip(lower=train_bounds.loc["min", col], upper=train_bounds.loc["max", col])

    # undo log transforms
    for col in ["KSOL", "HLM CLint","MLM CLint","Caco-2 Permeability Papp A>B","Caco-2 Permeability Efflux","MPPB","MBPB","MGMB"]:
        df[col] = np.expm1(df[col])
    df["Molecule Name"] = test_df["Molecule Name"]
    df["SMILES"] = test_df["SMILES"]
    df[["Molecule Name", "SMILES", "LogD","KSOL","HLM CLint","MLM CLint","Caco-2 Permeability Papp A>B","Caco-2 Permeability Efflux","MPPB","MBPB","MGMB"]].to_csv(
        f"test_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        index=False,
    )

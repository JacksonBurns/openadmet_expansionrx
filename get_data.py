import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover


def clean_smiles(
    smiles: str, remove_hs: bool = True, strip_stereochem: bool = False, strip_salts: bool = True
) -> str:
    """Applies preprocessing to SMILES strings, seeking the 'parent' SMILES

    Note that this is different from simply _neutralizing_ the input SMILES - we attempt to get the parent molecule, analogous to a molecular skeleton.
    This is adapted in part from https://rdkit.org/docs/Cookbook.html#neutralizing-molecules

    Args:
        smiles (str): input SMILES
        remove_hs (bool, optional): Removes hydrogens. Defaults to True.
        strip_stereochem (bool, optional): Remove R/S and cis/trans stereochemistry. Defaults to False.
        strip_salts (bool, optional): Remove salt ions. Defaults to True.

    Returns:
        str: cleaned SMILES
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None, f"Could not parse SMILES {smiles}"
        if remove_hs:
            mol = Chem.RemoveHs(mol)
        if strip_stereochem:
            Chem.RemoveStereochemistry(mol)
        if strip_salts:
            remover = SaltRemover()  # use default saltremover
            mol = remover.StripMol(mol)  # strip salts

        pattern = Chem.MolFromSmarts("[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]")
        at_matches = mol.GetSubstructMatches(pattern)
        at_matches_list = [y[0] for y in at_matches]
        if len(at_matches_list) > 0:
            for at_idx in at_matches_list:
                atom = mol.GetAtomWithIdx(at_idx)
                chg = atom.GetFormalCharge()
                hcount = atom.GetTotalNumHs()
                atom.SetFormalCharge(0)
                atom.SetNumExplicitHs(hcount - chg)
                atom.UpdatePropertyCache()
        out_smi = Chem.MolToSmiles(mol, kekuleSmiles=True)  # this also canonicalizes the input
        assert len(out_smi) > 0, f"Could not convert molecule to SMILES {smiles}"
        return out_smi
    except Exception as e:
        print(f"Failed to clean SMILES {smiles} due to {e}")
        return None


if __name__ == "__main__":
    # training
    df = pd.read_csv(
        "hf://datasets/openadmet/openadmet-expansionrx-challenge-train-data/expansion_data_train_raw.csv"
    )
    # clip the highest 5% of values for each target to reduce the impact of extreme outliers
    for target in [
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
        upper_bound = df[target].quantile(0.95)
        df.loc[df[target] > upper_bound, target] = upper_bound
    # log transform certain targets
    for target in [
        "KSOL",
        "HLM CLint",
        "MLM CLint",
        "Caco-2 Permeability Papp A>B",
        "Caco-2 Permeability Efflux",
        "MPPB",
        "MBPB",
        "MGMB",
    ]:
        df[target] = np.log1p(df[target])
    df["clean_smiles"] = df["SMILES"].apply(clean_smiles)
    df.to_csv("train.csv", index=False)

    # testing
    df = pd.read_csv(
        "hf://datasets/openadmet/openadmet-expansionrx-challenge-test-data-blinded/expansion_data_test_blinded.csv"
    )
    df["clean_smiles"] = df["SMILES"].apply(clean_smiles)
    df.to_csv("test.csv", index=False)

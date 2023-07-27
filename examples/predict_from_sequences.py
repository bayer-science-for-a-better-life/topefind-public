"""
Simple python script that shows how to predict the paratopes with sequence-based, unpaired methods.
Defines some models to use, trains the classification head, and then predict!
"""
import numpy as np
import pandas as pd
from joblib import Memory
from tabulate import tabulate
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestClassifier

from topefind.data_hub import SabdabHub
from topefind.predictors import (
    PLMSKClassifier,
    Parapred,
    EndToEndPredictorName,
)
from topefind.embedders import (
    PhysicalPropertiesPosContextEmbedder,
    ESMEmbedder,
    EmbedderName,
)

from topefind.utils import (
    SABDAB_PATH,
    VALID_IMGT_IDS,
    rescale,
)

PRIMARY_KEY = [
    "antibody_sequence",
    "antibody_chain",
    "chain_type",
    "antigen_chain",
]
INTERESTED_COLUMNS = [
    "pdb",
    "antibody_sequence",
    "antibody_imgt",
    "antibody_chain",
    "chain_type",
    "resolution",
    "scfv",
    "antigen_sequence",
    "antigen_chain",
    "antigen_type",
    "num_antigen_chains",
    "paratope_labels",
    "full_paratope_labels",
]
N_JOBS = 4
N_ESTIMATORS = 100
MAX_IMGT_ID = VALID_IMGT_IDS[-1]
MEMORY = Memory("cache", verbose=0)
SEED = 42


# ----------------------------------------------------------------------------------------------------------------------
# DEFINE MODELS
# ----------------------------------------------------------------------------------------------------------------------
# Some models need to be cached if further used on the same training dataset.
# Feel free to switch to ESM2 bigger models available under EmbedderName.
# Check `topefind.predictors` and `topefind.embedders`and choose your desired configuration.
@MEMORY.cache
def esm_rf(
        train_sequences,
        train_labels,
        emb_name=EmbedderName.esm2_8m,
        n_estimators=N_ESTIMATORS
):
    return PLMSKClassifier(
        ESMEmbedder(emb_name),
        RandomForestClassifier(
            n_estimators=n_estimators,
            n_jobs=N_JOBS,
            random_state=42
        )
    ).train(train_sequences, train_labels)


@MEMORY.cache
def simple_features_rf(
        train_sequences,
        train_labels,
        preprocessed_df=None,
        emb_name=EmbedderName.imgt_aa_ctx_23,
        n_estimators=N_ESTIMATORS,
):
    return PLMSKClassifier(
        PhysicalPropertiesPosContextEmbedder(
            emb_name,
            imgts=VALID_IMGT_IDS,
            precomputed_imgts_df=preprocessed_df
        ),
        RandomForestClassifier(
            n_estimators=n_estimators,
            n_jobs=N_JOBS,
            random_state=42
        )
    ).train(train_sequences, train_labels)


def main():
    # ------------------------------------------------------------------------------------------------------------------
    # GET DATA
    # ------------------------------------------------------------------------------------------------------------------
    # Load the curated, labelled, compact dataset derived from SAbDab.
    df: pd.DataFrame = pd.read_parquet(SABDAB_PATH / "sabdab.parquet", columns=INTERESTED_COLUMNS)

    # Let's filter according to literature guidelines.
    df = df.drop_duplicates("antibody_sequence")  # Don't bias the model.
    df = df[(df.antibody_sequence.str.len() > 70) & (df.antibody_sequence.str.len() < 200)]  # Don't go < 70 ...ANARCI.
    df = df[df.full_paratope_labels.apply(sum) >= 1]  # At least some positives.
    df = df[(df.num_antigen_chains > 0) & (df.num_antigen_chains <= 3)]  # Follows the choice in Paragraph.
    df = df[~df.scfv]  # Hard to deal with since two chains are connected.
    df = df[df.antigen_type.isin([
        "protein",
        "peptide",
        "protein | protein",
        "protein | peptide",
        "peptide | protein",
        "peptide | peptide",
    ])]
    df = df[df.resolution < 3]  # Allows to define contacts above this resolution (used everywhere in literature).
    df = df.reset_index()

    # Now the set of unique ["antibody_sequence", "antibody_chain", "chain_type", "antigen_chain"]
    # needs to be equal to the total number of elements in the DataFrame.
    assert len(np.unique(["_".join(el) for el in df[PRIMARY_KEY].values])) == len(df), "Mismatch in primary key"
    print(df.head())

    df = SabdabHub.fix_imgts(df)

    # Done, a working dataset.
    print(f"Dataset now contains {len(df)} entries")
    print(f"{len(df[df.num_antigen_chains > 1])} entries are connected to multiple antigens")

    all_df = df
    df = df.sample(frac=0.5, replace=False, random_state=42)
    sequences = df["antibody_sequence"].to_list()
    labels = df["full_paratope_labels"].to_list()

    # Keep "a couple out" (alla sarda) to test upon.
    test_size = 10
    train_sequences, train_labels = sequences[:-test_size], labels[:-test_size]
    test_sequences, test_labels = sequences[-test_size:], labels[-test_size:]

    # ------------------------------------------------------------------------------------------------------------------
    # TRAIN
    # ------------------------------------------------------------------------------------------------------------------
    # Let's instantiate some models.
    # Remember to pass a DF that contains at least "antigen_sequence" and "antigen_imgt"
    # if you do not want to use ANARCI to regenerate the numbering.
    parapred = Parapred(EndToEndPredictorName.parapred)
    esm = esm_rf(train_sequences, train_labels)
    simple_model = simple_features_rf(train_sequences, train_labels, preprocessed_df=all_df)

    # ------------------------------------------------------------------------------------------------------------------
    # SHOW RESULTS
    # ------------------------------------------------------------------------------------------------------------------
    candidate = test_sequences[0]
    ground_truth = test_labels[0]
    models = [parapred, esm, simple_model]
    models_names = [model.name for model in models]
    preds = [model.predict(candidate) for model in models]
    scaled_preds = [rescale(pred) for pred in preds]  # Scaling the prediction across the antibody
    ranked_preds = [rankdata(pred) for pred in preds]  # Ranking the prediction across the antibody

    dict_results = {"sequence": [aa for aa in candidate]} \
                   | {"ground_truth": [aa for aa in ground_truth]} \
                   | {f"{name}_preds": preds[i] for i, name in enumerate(models_names)} \
                   | {f"{name}_scaled_preds": scaled_preds[i] for i, name in enumerate(models_names)} \
                   | {f"{name}_ranked_preds": ranked_preds[i] for i, name in enumerate(models_names)}
    results = pd.DataFrame(dict_results)
    print(tabulate(results, headers=results.columns, tablefmt="simple_grid", floatfmt=".2f"))


if __name__ == '__main__':
    main()

# Some remarks:

# Special regards to IMGT numbering computation.
# There are three ways to do it that come to my mind:
# 1) Adding manually exceptions, accounting for more positions.
# 2) Recomputing everything in ANARCI once again.
# 3) Use all possible valid IMGT positions and throw errors when something is out of place (or filter strange Abs).
#
# Currently, I am in support of 3). utils.py defines the valid IMGT positions.
#
# Here is how to do 2):
# @MEMORY.cache
# def re_compute_anarci(_df):
#     index_to_use = []
#     new_antibody_imgts = []
#     for i, seq in enumerate(_df["antibody_sequence"].to_numpy()):
#         numbering = get_antibody_numbering(seq, scheme="imgt")
#         if numbering is not None:
#             index_to_use.append(i)
#             new_antibody_imgts.append(numbering)
#         else:
#             print(f"ANARCI could not parse: {_df['pdb'].iloc[i]}")
#     return index_to_use, new_antibody_imgts

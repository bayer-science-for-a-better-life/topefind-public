# Be aware that this can be computationally heavy.
# Tweak N_JOBS according to your available hardware.

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from transformers import EsmModel, EsmConfig

from topefind.utils import get_device, TOPEFIND_PATH, NAME_BEAUTIFIER
from topefind.predictors import PLMSKClassifier, PosFrequency, Predictor
from topefind.embedders import (
    Embedder, ESMEmbedder, EmbedderName,
    PhysicalPropertiesNoPosEmbedder,
    PhysicalPropertiesPosEmbedder,
    PhysicalPropertiesPosContextEmbedder
)

from exploration import *

FILE_PATH = Path(__file__)
INTERESTED_COLUMNS = [
    "pdb",
    "antibody_chain",
    "chain_type",
    "antibody_sequence",
    "antibody_imgt",
    "paratope_labels",
    "full_paratope_labels",
    "antigen_sequence",
    "antigen_chain",
    "antigen_type",
    "num_antigen_chains",
    "resolution",
    "method",
    "scfv",
]
RANDOM_STATES = np.arange(3)
NUM_FRACTIONS = 20
N_ESTIMATORS = 128
N_JOBS = 64

# ESM2 8M config for random weights - untrained
CONFIG_ESM2_8M_RANDOM = EsmConfig(
    vocab_size=33, mask_token_id=32, pad_token_id=1, hidden_size=320,
    num_hidden_layers=6, num_attention_heads=20, intermediate_size=1280,
    hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0, layer_norm_eps=1e-05,
    position_embedding_type="rotary", token_dropout=True, model_type="esm",
    hidden_act="gelu", emb_layer_norm_before=False, torch_dtype="float32",
    output_hidden_states=True, max_position_embeddings=1026
)

# ESM2 650M config for random weights - untrained
CONFIG_ESM2_650M_RANDOM = EsmConfig(
    vocab_size=33, mask_token_id=32, pad_token_id=1, hidden_size=1280,
    num_hidden_layers=33, num_attention_heads=20, intermediate_size=5120,
    hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0, layer_norm_eps=1e-05,
    position_embedding_type="rotary", token_dropout=True, model_type="esm",
    hidden_act="gelu", emb_layer_norm_before=False, torch_dtype="float32",
    output_hidden_states=True
)

TRAIN = pl.scan_parquet(TOPEFIND_PATH.parent / "resources/paragraph_train.parquet") \
    .select(INTERESTED_COLUMNS) \
    .collect()

TEST = pl.scan_parquet(TOPEFIND_PATH.parent / "resources/paragraph_test.parquet") \
    .select(INTERESTED_COLUMNS) \
    .collect()

TRAIN_SEQUENCES = TRAIN.get_column("antibody_sequence").to_list()
TRAIN_LABELS = TRAIN.get_column("full_paratope_labels").to_list()

TEST_SEQUENCES = TEST.get_column("antibody_sequence").to_list()
TEST_LABELS = TEST.get_column("full_paratope_labels").to_list()


def clean_mem(device, model):
    del model
    if "cuda" in device:
        torch.cuda.empty_cache()


def increase_train_size(embedder: Embedder):
    ap_means = []
    ap_stds = []
    models = []
    r_states = []
    fraction_values = []

    for random_state in RANDOM_STATES:
        train_df = TRAIN.sample(frac=1.0, seed=random_state)
        fractions = (np.arange(NUM_FRACTIONS) + 1) / NUM_FRACTIONS

        for frac in fractions:
            print(f"Random state: {random_state} - Current training size {frac}")
            current_num_samples = int(len(train_df) * frac)

            model = PLMSKClassifier(
                embedder=embedder,
                classifier=RandomForestClassifier(
                    n_estimators=N_ESTIMATORS,
                    n_jobs=N_JOBS,
                    random_state=random_state
                ),
            )

            X_train, y_train = model.prepare_dataset(train_df[:current_num_samples])
            model.train(X_train, y_train)
            scores = [average_precision_score(yt, model.predict(seq))
                      for yt, seq in zip(TEST_LABELS, TEST_SEQUENCES)]

            ap_means.append(np.mean(scores))
            ap_stds.append(np.std(scores))
            models.append(str(model.name))
            r_states.append(random_state)
            fraction_values.append(frac)
            print(f"AP: {ap_means[-1]:.2F} +/- {ap_stds[-1]:.2F}")

    return pd.DataFrame({
        "mean_ap": ap_means,
        "std_ap": ap_stds,
        "model": models,
        "r_state": r_states,
        "fraction": fraction_values
    })


def increase_train_size_predictor_only(predictor):
    ap_means = []
    ap_stds = []
    models = []
    r_states = []
    fraction_values = []

    for random_state in RANDOM_STATES:
        train_df = TRAIN.sample(frac=1.0, seed=random_state)
        fractions = (np.arange(NUM_FRACTIONS) + 1) / NUM_FRACTIONS

        for frac in fractions:
            print(f"Random state: {random_state} - Current training size {frac}")
            current_num_samples = int(len(train_df) * frac)

            X_train, y_train = predictor.prepare_dataset(train_df[:current_num_samples])
            predictor.train(X_train, y_train)

            scores = []
            for yt, seq in zip(TEST_LABELS, TEST_SEQUENCES):
                try:
                    scores.append(average_precision_score(yt, predictor.predict(seq)))
                except ValueError:
                    print("Mismatch of lengths or additional errors")

            ap_means.append(np.mean(scores))
            ap_stds.append(np.std(scores))
            models.append(str(predictor.name))
            r_states.append(random_state)
            fraction_values.append(frac)
            print(f"AP: {ap_means[-1]:.2F} +/- {ap_stds[-1]:.2F}")

    return pd.DataFrame({
        "mean_ap": ap_means,
        "std_ap": ap_stds,
        "model": models,
        "r_state": r_states,
        "fraction": fraction_values
    })


def run_experiment():
    device = get_device()

    model = ESMEmbedder(EmbedderName.esm2_8m, device=device)
    esm2_8m_res = increase_train_size(model)
    clean_mem(device, model)

    model = ESMEmbedder(EmbedderName.esm2_650m, device=device)
    esm2_650m_res = increase_train_size(model)
    clean_mem(device, model)

    esm2_8m_untrained = ESMEmbedder(EmbedderName.esm2_8m)
    esm2_8m_untrained.model = EsmModel(config=CONFIG_ESM2_8M_RANDOM).to(device)
    esm2_8m_untrained.name += "_untrained"
    esm2_8m_untrained_res = increase_train_size(esm2_8m_untrained)
    clean_mem(device, esm2_8m_untrained)

    esm2_650m_untrained = ESMEmbedder(EmbedderName.esm2_650m)
    esm2_650m_untrained.model = EsmModel(config=CONFIG_ESM2_650M_RANDOM).to(device)
    esm2_650m_untrained.name += "_untrained"
    esm2_650m_untrained_res = increase_train_size(esm2_650m_untrained)
    clean_mem(device, esm2_650m_untrained)

    df = pd.concat([
        esm2_8m_untrained_res,
        esm2_650m_untrained_res,
        esm2_8m_res,
        esm2_650m_res,
        increase_train_size(PhysicalPropertiesNoPosEmbedder(EmbedderName.aa)),
        increase_train_size(PhysicalPropertiesPosEmbedder(EmbedderName.imgt_aa)),
        increase_train_size(PhysicalPropertiesPosContextEmbedder(EmbedderName.imgt_aa_ctx_23)),
        increase_train_size_predictor_only(PosFrequency())
    ])

    df.to_csv(FILE_PATH.parent / "training_size.csv")


if __name__ == '__main__':
    # run_experiment()

    df = pd.read_csv("training_size.csv")
    df = df[df["model"] != "aa"]
    df = df[df["model"] != "imgt_aa"]
    df["model"] = df["model"].apply(
        lambda x: NAME_BEAUTIFIER[x + "_rf"] if "pos_freq" not in x else NAME_BEAUTIFIER[x]
    )

    fig, axs = plt.subplots(1, 1, figsize=(6, 5))
    sns.lineplot(
        df,
        x="fraction",
        y="mean_ap",
        hue="model",
        style="model",
        markers=True,
        dashes=False,
        ax=axs
    )
    axs.set_title("Performance and Training Size", fontsize=FONTSIZE_TITLE)
    axs.set_xlabel("Fraction of the training set", fontsize=FONTSIZE_LABELS)
    axs.set_ylabel("AP", fontsize=FONTSIZE_LABELS)
    axs.set_xlim(0.00, 1.10)
    axs.set_ylim(0.22, 0.80)
    # axs.grid()
    axs.legend(loc="center", fontsize="8")
    sns.despine(ax=axs)

    plt.tight_layout()
    plt.savefig(FILE_PATH.parent / "training_size.pdf")

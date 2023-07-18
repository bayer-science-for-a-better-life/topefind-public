"""
Let's find the best hyperparameters for a Random Forest.
This script is easily expandable to perform model selection as well.
(Previous results, however, show that Random Forest overcomes other methods (HGB, LR, NN, CNN))
"""

import numpy as np
import polars as pl
from tqdm import tqdm
from sklearn.metrics import average_precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from topefind.embedders import ESMEmbedder, EmbedderName
from topefind.utils import TOPEFIND_PATH

EMBEDDER = ESMEmbedder(EmbedderName.esm2_8m)
SEED = 42
N_FOLDS = 3

# ----------------------------------------------------------------------------------------------------------------------
# GET DATA
# ----------------------------------------------------------------------------------------------------------------------
train_df = pl.read_parquet(TOPEFIND_PATH / "exploration/out/paragraph_train.parquet")
test_df = pl.read_parquet(TOPEFIND_PATH / "exploration/out/paragraph_test.parquet")

train_df = train_df.sample(frac=1, seed=SEED).to_pandas()
test_df = test_df.sample(frac=1, seed=SEED).to_pandas()

seqs_train = train_df["antibody_sequence"].to_list()
seqs_test = test_df["antibody_sequence"].to_list()

labels_train = train_df["full_paratope_labels"].to_list()
labels_test = test_df["full_paratope_labels"].to_list()

# ----------------------------------------------------------------------------------------------------------------------
# GET EMBEDDINGS
# ----------------------------------------------------------------------------------------------------------------------
y_train = np.concatenate(labels_train)
y_test = np.concatenate(labels_test)
X_train = np.vstack([EMBEDDER.embed(inputs)[0].to("cpu").numpy() for inputs in tqdm(seqs_train)]).astype(float)
X_test = np.vstack([EMBEDDER.embed(inputs)[0].to("cpu").numpy() for inputs in tqdm(seqs_test)]).astype(float)

print(f"Training set has {X_train.shape[0]} elements, with {X_train.shape[1]} features each")
print(f"Test set has {X_test.shape[0]} elements, with {X_test.shape[1]} features each")

# ----------------------------------------------------------------------------------------------------------------------
# SET UP HYPERPARAMETER TUNING
# ----------------------------------------------------------------------------------------------------------------------
rfc_param_grid = [{
    "max_depth": [5, 10, 25, 50, None],
    "n_estimators": [128, 256]
}]
models = {
    "rfc": {
        "name": "RF       ",
        "estimator": RandomForestClassifier(random_state=SEED, n_jobs=32),
        "param": rfc_param_grid
    },
}

results_short_val = {}
results_short_test = {}

# ----------------------------------------------------------------------------------------------------------------------
# TUNE
# ----------------------------------------------------------------------------------------------------------------------
for clf in models.keys():
    print(f"\nTrying model: {models[clf]['name']}")
    print("------------------------------------------")
    model = GridSearchCV(
        models[clf]["estimator"],
        models[clf]["param"],
        cv=StratifiedKFold(N_FOLDS),
        n_jobs=5,
        scoring="average_precision",

    )
    model.fit(X_train, y_train)

    print(f"Best parameters set found:\n{model.best_params_}\n")
    print("Grid scores on the validation sets: ")

    means_cv = model.cv_results_["mean_test_score"]  # over CV val sets, don't confuse with holdout set
    stds_cv = model.cv_results_["std_test_score"]  # over CV val sets, don't confuse with holdout set
    params = model.cv_results_["params"]

    for mean, std, params_tuple in zip(means_cv, stds_cv, params):
        print(f"{mean:.2F} (+/-{std:.2F}) over {N_FOLDS} folds for {params_tuple}")

    # Saving the score on entire GridSearchCV
    results_short_val[clf] = model.best_score_
    y_pred = model.predict_proba(X_test)[:, -1]
    results_short_test[clf] = average_precision_score(y_test, y_pred)
    print("------------------------------------------\n")

# ----------------------------------------------------------------------------------------------------------------------
# SHOW RESULTS
# ----------------------------------------------------------------------------------------------------------------------
# Pay attention, the results are given on the whole test and validation sets and not on each antibody here.
print("\nSummary of best results on validation set:")
print("Estimator")
for clf in results_short_val.keys():
    print(f"{models[clf]['name']}\t - score: {results_short_val[clf]:.3F}")

print("\nSummary of best results on the test set:")
print("Estimator")
for clf in results_short_test.keys():
    print(f"{models[clf]['name']}\t - score: {results_short_test[clf]:.3F}")

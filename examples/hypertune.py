"""
Let's find the best hyperparameters and a classifier for ESM embeddings.
(Results show that Random Forest overcomes other methods (HGB, LR, NN, KNN))
"""

import numpy as np
import polars as pl
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
from sklearn.metrics import average_precision_score
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from topefind.embedders import ESMEmbedder, EmbedderName
from topefind.utils import TOPEFIND_PATH

EMBEDDER = ESMEmbedder(EmbedderName.esm2_8m)
SEED = 42
N_FOLDS = 3

# ----------------------------------------------------------------------------------------------------------------------
# GET DATA
# ----------------------------------------------------------------------------------------------------------------------
train_df = pl.read_parquet(TOPEFIND_PATH.parent / "resources/paragraph_train.parquet")
test_df = pl.read_parquet(TOPEFIND_PATH.parent / "resources/paragraph_test.parquet")

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

train_embs = [EMBEDDER.embed(inputs)[0].to("cpu").numpy() for inputs in tqdm(seqs_train)]
test_embs = [EMBEDDER.embed(inputs)[0].to("cpu").numpy() for inputs in tqdm(seqs_test)]
X_train = np.vstack(train_embs).astype(float)
X_test = np.vstack(test_embs).astype(float)

print(f"Training set has {X_train.shape[0]} elements, with {X_train.shape[1]} features each")
print(f"Test set has {X_test.shape[0]} elements, with {X_test.shape[1]} features each")

# ----------------------------------------------------------------------------------------------------------------------
# SET UP HYPERPARAMETER TUNING
# ----------------------------------------------------------------------------------------------------------------------
rfc_param_grid = [{
    "max_depth": [5, 10, 25, 50, None],
    "n_estimators": [128, 256],
    "class_weight": ["balanced", None]
}]
dt_param_grid = [{
    "max_depth": [5, 10, 25, 50, None],
    "class_weight": ["balanced", None]
}]
hgb_param_grid = [{
    "max_iter": [100, 200, 300],
}]
lr_param_grid = [{
    "max_iter": [100, 200, 300],
}]
mlp_param_grid = [{
    "max_iter": [100, 200, 300],
    "batch_size": [128],
    "hidden_layer_sizes": [(128, 64, 32,), (64, 32,), (32,)],
}]
knn_grid = [{
    "n_neighbors": [5, 10, 15, 20, 50, 100],
}]

models = {
    "rfc": {
        "name": "RF       ",
        "estimator": RandomForestClassifier(random_state=SEED, n_jobs=32),
        "param": rfc_param_grid
    },
    "dt": {
        "name": "DT       ",
        "estimator": DecisionTreeClassifier(random_state=SEED),
        "param": dt_param_grid
    },
    "hgb": {
        "name": "HGB      ",
        "estimator": HistGradientBoostingClassifier(random_state=SEED),
        "param": hgb_param_grid
    },
    "lr": {
        "name": "LR       ",
        "estimator": LogisticRegression(random_state=SEED, n_jobs=32),
        "param": lr_param_grid
    },
    "mlp": {
        "name": "MLP      ",
        "estimator": MLPClassifier(random_state=SEED),
        "param": mlp_param_grid
    },
    "knn": {
        "name": "KNN      ",
        "estimator": KNeighborsClassifier(),
        "param": knn_grid
    },
}

# results_short_train = {} # TODO: find indexes from GridSearch
results_short_val = {}
results_short_test = {}

results_over_ab_train = {}
# results_over_ab_val = {} # TODO: find indexes from GridSearch
results_over_ab_test = {}

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
        refit=True,
        verbose=1,
    )
    model.fit(X_train, y_train)

    print(f"Best parameters set found:\n{model.best_params_}\n")
    print("Grid scores on the validation sets: ")

    means_cv_val = model.cv_results_["mean_test_score"]  # over CV val sets, don't confuse with holdout set
    stds_cv_val = model.cv_results_["std_test_score"]  # over CV val sets, don't confuse with holdout set

    params = model.cv_results_["params"]

    for mean_val, std_val, params_tuple in zip(means_cv_val, stds_cv_val, params):
        print(f"Val: {mean_val:.2F} (+/-{std_val:.2F}) over {N_FOLDS} folds for {params_tuple}")

    # Saving the score on entire GridSearchCV
    results_short_val[clf] = model.best_score_
    y_pred = model.predict_proba(X_test)[:, -1]
    results_short_test[clf] = average_precision_score(y_test, y_pred)

    # Saving mean scores of antibodies
    ab_train_preds = [model.predict_proba(ab_embs)[:, -1] for ab_embs in train_embs]
    ab_test_preds = [model.predict_proba(ab_embs)[:, -1] for ab_embs in test_embs]

    train_scores = [average_precision_score(lab, pred) for lab, pred in zip(labels_train, ab_train_preds)]
    test_scores = [average_precision_score(lab, pred) for lab, pred in zip(labels_test, ab_test_preds)]
    results_over_ab_train[clf] = f"{np.mean(train_scores):.3F} +/- {np.std(train_scores):.3F}"
    results_over_ab_test[clf] = f"{np.mean(test_scores):.3F} +/- {np.std(test_scores):.3F}"
    print("------------------------------------------\n")

# ----------------------------------------------------------------------------------------------------------------------
# SHOW RESULTS
# ----------------------------------------------------------------------------------------------------------------------
# Pay attention, the results are given on the whole test and validation sets and not on each antibody here.
print("\nSummary of best results on train set:")
print("Estimator")
for clf in results_over_ab_train.keys():
    print(f"Antibody-wise: {models[clf]['name']}\t - score: {results_over_ab_train[clf]}")

print("\nSummary of best results on validation set:")
print("Estimator")
for clf in results_short_val.keys():
    print(f"Combined: {models[clf]['name']}\t - score: {results_short_val[clf]:.3F}")

print("\nSummary of best results on the test set:")
print("Estimator")
for clf in results_short_test.keys():
    print(f"Combined: {models[clf]['name']}\t - score: {results_short_test[clf]:.3F}")
    print(f"Antibody-wise: {models[clf]['name']}\t - score: {results_over_ab_test[clf]}")

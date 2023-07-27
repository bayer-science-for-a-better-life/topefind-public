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
    "class_weights": ["balanced", None]
}]
dt_param_grid = [{
    "max_depth": [5, 10, 25, 50, None],
    "class_weights": ["balanced", None]
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
    )
    model.fit(X_train, y_train)

    print(f"Best parameters set found:\n{model.best_params_}\n")
    print("Grid scores on the validation sets: ")

    means_cv_train = model.cv_results_["mean_train_score"]
    stds_cv_train = model.cv_results_["std_train_score"]
    means_cv_val = model.cv_results_["mean_test_score"]  # over CV val sets, don't confuse with holdout set
    stds_cv_val = model.cv_results_["std_test_score"]  # over CV val sets, don't confuse with holdout set

    params = model.cv_results_["params"]

    for mean_train, std_train, mean_val, std_val, params_tuple in \
            zip(means_cv_train, stds_cv_train, means_cv_val, stds_cv_val, params):

        print(f"Train: {mean_train:.2F} (+/-{std_train:.2F}) | "
              f"Val: {mean_train:.2F} (+/-{std_val:.2F}) "
              f"over {N_FOLDS} folds for {params_tuple}")

    # Saving the score on entire GridSearchCV
    results_short_val[clf] = model.best_score_
    y_pred = model.predict_proba(X_test)[:, -1]
    results_short_test[clf] = average_precision_score(y_test, y_pred)

    # Saving mean scores of antibodies
    ab_train_preds = [model.predict_proba(ab_embs)[:, -1] for ab_embs in train_embs]
    ab_test_preds = [model.predict_proba(ab_embs)[:, -1] for ab_embs in test_embs]

    train_scores = [average_precision_score(lab, pred) for lab, pred in zip(labels_train, ab_train_preds)]
    test_scores = [average_precision_score(lab, pred) for lab, pred in zip(labels_test, ab_test_preds)]
    results_over_ab_train[clf] = f"{np.mean(train_scores)} +/- {np.std(train_scores)}"
    results_over_ab_test[clf] = f"{np.mean(test_scores)} +/- {np.std(test_scores)}"
    print("------------------------------------------\n")

# ----------------------------------------------------------------------------------------------------------------------
# SHOW RESULTS
# ----------------------------------------------------------------------------------------------------------------------
# Pay attention, the results are given on the whole test and validation sets and not on each antibody here.
print("\nSummary of best results on train set:")
print("Estimator")
for clf in results_over_ab_train.keys():
    print(f"Antibody-wise: {models[clf]['name']}\t - score: {results_over_ab_train[clf]:.3F}")

print("\nSummary of best results on validation set:")
print("Estimator")
for clf in results_short_val.keys():
    print(f"Combined: {models[clf]['name']}\t - score: {results_short_val[clf]:.3F}")

print("\nSummary of best results on the test set:")
print("Estimator")
for clf in results_short_test.keys():
    print(f"Combined: {models[clf]['name']}\t - score: {results_short_test[clf]:.3F}")
    print(f"Antibody-wise: {models[clf]['name']}\t - score: {results_over_ab_test[clf]:.3F}")

# ----------------------------------------------------------------------------------------------------------------------
# OUTPUT
# ----------------------------------------------------------------------------------------------------------------------

"""
100%|█████████████████████████████████████████████| 1289/1289 [00:38<00:00, 33.54it/s]
100%|██████████████████████████████████████████████| 359/359 [00:01<00:00, 188.61it/s]
Training set has 146787 elements, with 320 features each
Test set has 40835 elements, with 320 features each

Trying model: RF       
------------------------------------------
Best parameters set found:
{'max_depth': None, 'n_estimators': 256}

Grid scores on the validation sets: 
0.53 (+/-0.01) over 3 folds for {'max_depth': 5, 'n_estimators': 128}
0.54 (+/-0.01) over 3 folds for {'max_depth': 5, 'n_estimators': 256}
0.62 (+/-0.02) over 3 folds for {'max_depth': 10, 'n_estimators': 128}
0.62 (+/-0.02) over 3 folds for {'max_depth': 10, 'n_estimators': 256}
0.65 (+/-0.02) over 3 folds for {'max_depth': 25, 'n_estimators': 128}
0.66 (+/-0.02) over 3 folds for {'max_depth': 25, 'n_estimators': 256}
0.65 (+/-0.02) over 3 folds for {'max_depth': 50, 'n_estimators': 128}
0.66 (+/-0.02) over 3 folds for {'max_depth': 50, 'n_estimators': 256}
0.65 (+/-0.02) over 3 folds for {'max_depth': None, 'n_estimators': 128}
0.66 (+/-0.02) over 3 folds for {'max_depth': None, 'n_estimators': 256}
------------------------------------------


Trying model: HGB      
------------------------------------------
Best parameters set found:
{'max_iter': 300}

Grid scores on the validation sets: 
0.64 (+/-0.01) over 3 folds for {'max_iter': 100}
0.65 (+/-0.01) over 3 folds for {'max_iter': 200}
0.65 (+/-0.01) over 3 folds for {'max_iter': 300}
------------------------------------------


Trying model: LR       
------------------------------------------
Best parameters set found:
{'max_iter': 300}

Grid scores on the validation sets: 
0.61 (+/-0.01) over 3 folds for {'max_iter': 100}
0.61 (+/-0.01) over 3 folds for {'max_iter': 200}
0.61 (+/-0.01) over 3 folds for {'max_iter': 300}
------------------------------------------


Trying model: MLP      
------------------------------------------
Best parameters set found:
{'batch_size': 128, 'hidden_layer_sizes': (32,), 'max_iter': 100}

Grid scores on the validation sets: 
0.58 (+/-0.02) over 3 folds for {'batch_size': 128, 'hidden_layer_sizes': (128, 64, 32), 'max_iter': 100}
0.58 (+/-0.02) over 3 folds for {'batch_size': 128, 'hidden_layer_sizes': (128, 64, 32), 'max_iter': 200}
0.58 (+/-0.02) over 3 folds for {'batch_size': 128, 'hidden_layer_sizes': (128, 64, 32), 'max_iter': 300}
0.58 (+/-0.02) over 3 folds for {'batch_size': 128, 'hidden_layer_sizes': (64, 32), 'max_iter': 100}
0.57 (+/-0.01) over 3 folds for {'batch_size': 128, 'hidden_layer_sizes': (64, 32), 'max_iter': 200}
0.57 (+/-0.01) over 3 folds for {'batch_size': 128, 'hidden_layer_sizes': (64, 32), 'max_iter': 300}
0.62 (+/-0.01) over 3 folds for {'batch_size': 128, 'hidden_layer_sizes': (32,), 'max_iter': 100}
0.60 (+/-0.02) over 3 folds for {'batch_size': 128, 'hidden_layer_sizes': (32,), 'max_iter': 200}
0.58 (+/-0.01) over 3 folds for {'batch_size': 128, 'hidden_layer_sizes': (32,), 'max_iter': 300}
------------------------------------------


Trying model: KNN      
------------------------------------------
Best parameters set found:
{'n_neighbors': 50}

Grid scores on the validation sets: 
0.57 (+/-0.01) over 3 folds for {'n_neighbors': 5}
0.61 (+/-0.01) over 3 folds for {'n_neighbors': 10}
0.63 (+/-0.01) over 3 folds for {'n_neighbors': 15}
0.64 (+/-0.01) over 3 folds for {'n_neighbors': 20}
0.64 (+/-0.01) over 3 folds for {'n_neighbors': 50}
0.63 (+/-0.01) over 3 folds for {'n_neighbors': 100}
------------------------------------------


Summary of best results on validation set:
Estimator
RF               - score: 0.656
HGB              - score: 0.647
LR               - score: 0.606
MLP              - score: 0.623
KNN              - score: 0.643

Summary of best results on the test set:
Estimator
RF               - score: 0.685
HGB              - score: 0.672
LR               - score: 0.605
MLP              - score: 0.663
KNN              - score: 0.651
"""

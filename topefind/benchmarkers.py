import click
import torch
import numpy as np
import polars as pl
from tabulate import tabulate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    matthews_corrcoef,
)

from topefind.data_hub import SabdabHub

from topefind.predictors import (
    Parapred,
    Paragraph,
    PLMSKClassifier,
    ContactsClassifier,
    Seq2ParatopeCDR,
    EndToEndPredictorName,
    Predictor,
    AAFrequency,
    PosFrequency,
    AAPosFrequency,
)

from topefind.embedders import (
    MultiChainType,
    EmbedderName,
    get_embedder_constructor,
)

from topefind.utils import (
    TOPEFIND_PATH,
    SABDAB_PATH,
    AB_REGIONS,
    VALID_IMGT_IDS,
    get_imgt_region_indexes,
    metric_at_top_k,
    iou,
    aiou,
)

MAX_IMGT_ID = VALID_IMGT_IDS[-1]
N_JOBS_TRAINING = 128
N_ESTIMATORS = 256
SEED = 42
SABDAB_PDBS_PATH = SABDAB_PATH / "all" / "imgt"
SABDAB_PROCESSED_PATH = SABDAB_PATH / "sabdab.parquet"
BENCHMARK_PATH = TOPEFIND_PATH.parent / "resources" / "benchmark.parquet"
DEVICE = "auto"

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

METRICS = [
    "ap",
    "iou",
    "mcc",
    "roc_auc",
    "precision",
    "recall",
    "f1",
    "bal_acc",
    "ap@5",
    "ap@10",
]

ACCEPTABLE_MODELS = [
    EndToEndPredictorName.parapred,
    EndToEndPredictorName.paragraph_untrained,
    EndToEndPredictorName.paragraph_unpaired,
    EndToEndPredictorName.paragraph_paired,
    EndToEndPredictorName.seq_to_cdr,
    EndToEndPredictorName.af2_multimer,
    EndToEndPredictorName.aa_freq,
    EndToEndPredictorName.pos_freq,
    EndToEndPredictorName.aa_pos_freq,
    EmbedderName.esm2_8m + "_rf",
    EmbedderName.esm2_35m + "_rf",
    EmbedderName.esm2_150m + "_rf",
    EmbedderName.esm2_650m + "_rf",
    EmbedderName.esm2_3b + "_rf",
    EmbedderName.esm1b + "_rf",
    EmbedderName.rita_s + "_rf",
    EmbedderName.rita_m + "_rf",
    EmbedderName.rita_l + "_rf",
    EmbedderName.rita_xl + "_rf",
    EmbedderName.prot_t5_xl + "_rf",
    EmbedderName.prot_t5_xxl + "_rf",
    EmbedderName.aa + "_rf",
    EmbedderName.imgt + "_rf",
    EmbedderName.imgt_aa + "_rf",
    EmbedderName.imgt_aa_ctx_3 + "_rf",
    EmbedderName.imgt_aa_ctx_5 + "_rf",
    EmbedderName.imgt_aa_ctx_7 + "_rf",
    EmbedderName.imgt_aa_ctx_11 + "_rf",
    EmbedderName.imgt_aa_ctx_17 + "_rf",
    EmbedderName.imgt_aa_ctx_23 + "_rf",
    EmbedderName.esm2_8m + "_" + MultiChainType.ag_aware + "_rf",
    EmbedderName.esm2_35m + "_" + MultiChainType.ag_aware + "_rf",
    EmbedderName.esm2_150m + "_" + MultiChainType.ag_aware + "_rf",
    EmbedderName.esm2_650m + "_" + MultiChainType.ag_aware + "_rf",
    EmbedderName.esm2_3b + "_" + MultiChainType.ag_aware + "_rf",
    EmbedderName.esm1b + "_" + MultiChainType.ag_aware + "_rf",
    EmbedderName.rita_s + "_" + MultiChainType.ag_aware + "_rf",
    EmbedderName.rita_m + "_" + MultiChainType.ag_aware + "_rf",
    EmbedderName.rita_l + "_" + MultiChainType.ag_aware + "_rf",
    EmbedderName.rita_xl + "_" + MultiChainType.ag_aware + "_rf",
    EmbedderName.prot_t5_xl + "_" + MultiChainType.ag_aware + "_rf",
    EmbedderName.prot_t5_xxl + "_" + MultiChainType.ag_aware + "_rf",
    EmbedderName.esm2_8m + "_" + MultiChainType.paired + "_rf",
    EmbedderName.esm2_35m + "_" + MultiChainType.paired + "_rf",
    EmbedderName.esm2_150m + "_" + MultiChainType.paired + "_rf",
    EmbedderName.esm2_650m + "_" + MultiChainType.paired + "_rf",
    EmbedderName.esm2_3b + "_" + MultiChainType.paired + "_rf",
    EmbedderName.esm1b + "_" + MultiChainType.paired + "_rf",
    EmbedderName.rita_s + "_" + MultiChainType.paired + "_rf",
    EmbedderName.rita_m + "_" + MultiChainType.paired + "_rf",
    EmbedderName.rita_l + "_" + MultiChainType.paired + "_rf",
    EmbedderName.rita_xl + "_" + MultiChainType.paired + "_rf",
    EmbedderName.prot_t5_xl + "_" + MultiChainType.paired + "_rf",
    EmbedderName.prot_t5_xxl + "_" + MultiChainType.paired + "_rf",
    EmbedderName.esm2_8m + "_de_biased_rf",
    EmbedderName.esm2_35m + "_de_biased_rf",
    EmbedderName.esm2_150m + "_de_biased_rf",
    EmbedderName.esm2_650m + "_de_biased_rf",
    EmbedderName.esm2_3b + "_de_biased_rf",
]


def prepare_embedder(
        train: pl.DataFrame,
        test: pl.DataFrame,
        name: str,
        device: str = "cpu",
        n_jobs: int = N_JOBS_TRAINING,
) -> PLMSKClassifier:
    embedder = get_embedder_constructor(name)
    stripped_emb_name = EmbedderName(
        name.
        removesuffix("_rf").
        removesuffix("_ag_aware").
        removesuffix("_paired").
        removesuffix("_de_biased")
    )

    if "_ag_aware" in name or "_paired" in name:
        base_embedder_constructor = get_embedder_constructor(stripped_emb_name)
        base_embedder = base_embedder_constructor(name=stripped_emb_name, device=device)
        if "_ag_aware" in name:
            embedder_model = embedder(base_embedder, MultiChainType.ag_aware)
        else:
            embedder_model = embedder(base_embedder, MultiChainType.paired)
    else:
        if "imgt" in name:
            # Let's pass to the model all the precomputed IMGTs so that we don't have to call ANARCI.
            df_with_imgts = pl.concat([train, test]).to_pandas()
            embedder_model = embedder(stripped_emb_name, device=device, precomputed_imgts_df=df_with_imgts)
        else:
            embedder_model = embedder(stripped_emb_name, device=device)

    clf = PLMSKClassifier(
        embedder_model,
        RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            n_jobs=n_jobs,
            random_state=SEED,
            # class_weight="balanced"
        ),
        de_bias="de_biased" in name
    )
    X_train, y_train = clf.prepare_dataset(train)
    clf.train(X_train, y_train)
    return clf


def prepare_testing(
        train: pl.DataFrame | None,
        test: pl.DataFrame | None,
        name: str,
        device: str,
        n_jobs: int = N_JOBS_TRAINING,
) -> tuple[list | None, list | None, Predictor]:
    # Order is important for correct parsing.
    # Paragraph needs to come before the embedder-based models.
    if "parapred" in name:
        model = Parapred(EndToEndPredictorName[name])
    elif "paragraph" in name:
        model = Paragraph(EndToEndPredictorName[name], SABDAB_PDBS_PATH)
    elif "seq_to_cdr" in name:
        model = Seq2ParatopeCDR(EndToEndPredictorName[name])
    elif "af2_multimer" in name:
        model = ContactsClassifier(EndToEndPredictorName[name])
    elif "aa_freq" in name:
        model = AAFrequency(EndToEndPredictorName[name])
        X, y = model.prepare_dataset(train)
        model.train(X, y)
    elif "pos_freq" == name:
        model = PosFrequency(EndToEndPredictorName[name])
        X, y = model.prepare_dataset(train)
        model.train(X, y)
    elif "aa_pos_freq" == name:
        model = AAPosFrequency(EndToEndPredictorName[name])
        X, y = model.prepare_dataset(train)
        model.train(X, y)
    else:
        model = prepare_embedder(train, test, name=name, device=device, n_jobs=n_jobs)

    X_test, y_test = model.prepare_dataset(test)
    return X_test, y_test, model


class ParagraphBenchmark:
    @staticmethod
    def preprocess_paragraph_csv(csv_path):
        df = pl.scan_csv(csv_path, has_header=False).select([
            pl.col("column_1").str.to_lowercase().alias("pdb").cast(pl.Utf8),
            pl.col("column_2").str.to_uppercase().alias("Lchain").cast(pl.Utf8),
            pl.col("column_3").str.to_uppercase().alias("Hchain").cast(pl.Utf8),
        ]).collect()
        h_pg = df.select(["pdb", "Hchain"]).rename({"Hchain": "antibody_chain"})
        l_pg = df.select(["pdb", "Lchain"]).rename({"Lchain": "antibody_chain"})
        pg_chain_types = pl.Series("chain_type", ["heavy"] * len(df) + ["light"] * len(df))
        df = pl.concat([h_pg, l_pg]).with_columns(pg_chain_types)
        return df

    @staticmethod
    def compute_metric(yt, yp, metric, thr=0.5):
        if metric == "ap":
            # Currently sklearn gives -0.0 if yt has only negative class.
            # https://github.com/scikit-learn/scikit-learn/issues/24381
            return max(average_precision_score(yt, yp), 0)
        elif metric == "roc_auc":
            return roc_auc_score(yt, yp)
        elif metric == "aiou":
            return aiou(yt, yp)
        else:
            yp_bin = np.where(yp < thr, 0, 1).astype(int)
            if metric == "iou":
                return iou(yt, yp_bin)
            elif metric == "mcc":
                return matthews_corrcoef(yt, yp_bin)
            elif metric == "precision":
                return precision_score(yt, yp_bin)
            elif metric == "recall":
                return recall_score(yt, yp_bin)
            elif metric == "f1":
                return f1_score(yt, yp_bin)
            elif metric == "bal_acc":
                return balanced_accuracy_score(yt, yp_bin)
            elif "ap@" in metric:
                return metric_at_top_k(yt, yp_bin, k=int(metric.split("@")[-1]))
            else:
                raise NotImplementedError(f"The following metric is not implemented: {metric}")

    @staticmethod
    def prepare_dataset():
        # Loading SAbDab interested data.
        sabdab = pl.scan_parquet(SABDAB_PROCESSED_PATH).select(INTERESTED_COLUMNS).collect()

        # Get paths to the train sets and the test set that we want to run the models on.
        parapred_train_set_path = TOPEFIND_PATH / "vendored/parapred/parapred/data/train_set.csv"
        paragraph_train_set_path = TOPEFIND_PATH / "vendored/Paragraph/training_data/Expanded/train_set.csv"
        paragraph_val_set_path = TOPEFIND_PATH / "vendored/Paragraph/training_data/Expanded/val_set.csv"
        paragraph_test_set_path = TOPEFIND_PATH / "vendored/Paragraph/training_data/Expanded/test_set.csv"

        # Get the DFs
        parapred_train_df = pl.scan_csv(parapred_train_set_path)
        parapred_train_df = parapred_train_df.select([
            pl.col("pdb").str.to_lowercase().cast(pl.Utf8),
            pl.col("Hchain").str.to_uppercase().cast(pl.Utf8),
            pl.col("Lchain").str.to_uppercase().cast(pl.Utf8),
        ]).collect()

        paragraph_train_df = ParagraphBenchmark.preprocess_paragraph_csv(paragraph_train_set_path)
        paragraph_val_df = ParagraphBenchmark.preprocess_paragraph_csv(paragraph_val_set_path)
        paragraph_test_df = ParagraphBenchmark.preprocess_paragraph_csv(paragraph_test_set_path)

        # We need to remove leakage... dataset.csv in parapred, contains pdb that are in the test file of paragraph.
        # The check is done only on the PDB id since only a few have different ids for H and L chains, and even in
        # that case, the antibody is a copy in the PDB file.
        join_pg_pp_val = paragraph_val_df.join(parapred_train_df, on="pdb").select("pdb").unique(subset="pdb")
        join_pg_pp_test = paragraph_test_df.join(parapred_train_df, on="pdb").select("pdb").unique(subset="pdb")
        join_pg_pg_val = paragraph_val_df.join(paragraph_train_df, on="pdb").select("pdb").unique(subset="pdb")
        join_pg_pg_test = paragraph_test_df.join(paragraph_train_df, on="pdb").select("pdb").unique(subset="pdb")

        splits_info = [
            (join_pg_pp_val, "Parapred train, Paragraph val", paragraph_val_df),
            (join_pg_pp_test, "Parapred train, Paragraph test", paragraph_test_df),
            (join_pg_pg_val, "Paragraph train, Paragraph val", paragraph_val_df),
            (join_pg_pg_test, "Paragraph train, Paragraph test", paragraph_test_df),
        ]

        for joined, splits_pair_name, tmp_df in splits_info:
            if len(joined) != 0:
                print(f"Train/test leakage between -> {splits_pair_name}")
                print(f"Removing {len(joined)} overlaps")

                to_tabulate = joined.sort(by="pdb").to_pandas()
                print(tabulate(to_tabulate, headers=["PDB Leakage"], tablefmt="latex", showindex=False))

                tmp_df = tmp_df.filter(~pl.col("pdb").is_in(joined.to_series()))
                print(f"Final size: {len(tmp_df)}, before: {len(tmp_df) + len(joined)}")
                if "Paragraph val" in splits_pair_name:
                    paragraph_val_df = paragraph_val_df.filter(~pl.col("pdb").is_in(joined.to_series()))
                else:
                    paragraph_test_df = paragraph_test_df.filter(~pl.col("pdb").is_in(joined.to_series()))
            else:
                print(f"No leakage between -> {splits_pair_name}")
        # Leakage is now removed, enjoy.

        # Let's train our classifiers on top of embeddings first.
        # Parse the training data, here we look only and the full paratope (multiple antigen chains might be involved)
        sabdab = sabdab.unique(subset=["pdb", "antibody_chain", "chain_type"])

        # Fix the IMGT and filter unwanted ones.
        sabdab = pl.DataFrame(SabdabHub.fix_imgts(sabdab.to_pandas()))

        train = sabdab.join(paragraph_train_df, on=["pdb", "antibody_chain", "chain_type"])
        val = sabdab.join(paragraph_val_df, on=["pdb", "antibody_chain", "chain_type"])
        test = sabdab.join(paragraph_test_df, on=["pdb", "antibody_chain", "chain_type"])

        # These can be saved for future usages.
        # train.write_parquet(TOPEFIND_PATH.parent / "resources/paragraph_train.parquet")
        # val.write_parquet(TOPEFIND_PATH.parent / "resources/paragraph_val.parquet")
        # test.write_parquet(TOPEFIND_PATH.parent / "resources/paragraph_test.parquet")
        return train, val, test

    @staticmethod
    @click.command()
    @click.option("--device", "-d", default=DEVICE, help="Device name, e.g. auto, cuda, cuda:0, cpu...")
    @click.option("--models", "-m", default=ACCEPTABLE_MODELS, help="Model name, which model to use.", multiple=True)
    @click.option("--save_path", "-sp", default=BENCHMARK_PATH, help="Where to save the results.")
    @click.option("--n_jobs", "-j", default=N_JOBS_TRAINING, help="Number of jobs for sklearn models.")
    def run(device, models, save_path, n_jobs):

        train, val, test = ParagraphBenchmark.prepare_dataset()
        if not isinstance(models, tuple):
            models = tuple(models)

        pdbs_test = test.get_column("pdb").to_list()
        chain_types_test = test.get_column("chain_type").to_list()
        ab_chains_test = test.get_column("antibody_chain").to_list()
        ab_seqs = test.get_column("antibody_sequence").to_list()
        ab_imgts = test.get_column("antibody_imgt").to_list()

        # First let's compute all the predictions.
        # Then let's calculate the metrics.
        ben_pdbs = []
        ben_chain_ids = []
        ben_chain_types = []
        ben_sequences = []
        ben_imgts = []
        ben_models = []
        ben_regions = []
        ben_scores = []
        ben_metrics = []
        ben_predictions = []
        ben_labels = []

        for model_name in models:
            assert model_name in ACCEPTABLE_MODELS, f"Passed model {model_name} is not supported"
            print(f"Evaluating {model_name}...")
            X_test, y_test, model = prepare_testing(train, test, model_name, device, n_jobs)

            zipped_digest = zip(pdbs_test, ab_chains_test, chain_types_test, ab_seqs, ab_imgts, X_test, y_test)
            for pdb, chain, chain_type, ab_seq, ab_imgt, xi, yi in zipped_digest:
                yp = np.array(model.predict(xi)).astype(float)
                yt = np.array(yi).astype(int)
                exploded_antibody_seq = np.array([aa for aa in ab_seq])

                for region_name in AB_REGIONS:
                    region_idxs = get_imgt_region_indexes(ab_imgt, region_name)
                    # Baselines based on numbering frequencies use ANARCI, and numbering might cause some sequences
                    # to differ. If you encounter this, you could (quickly and dirtly) use an exception for the
                    # next line and continue
                    try:
                        yp_region = yp[region_idxs]
                    except IndexError:
                        continue
                    yt_region = yt[region_idxs]
                    region_seq = "".join(exploded_antibody_seq[region_idxs])
                    ab_imgt_region = np.array(ab_imgt)[region_idxs]

                    if len(np.unique(yt_region)) != 2:
                        metrics_to_compute = ["ap", "bal_acc"]
                    elif region_name == "all":
                        metrics_to_compute = METRICS
                    else:
                        metrics_to_compute = ["ap", "iou", "bal_acc"]

                    # If you want to parallelize the metric computation, not sure if it would gain much more speed.
                    # metrics = Parallel(n_jobs=4)(
                    #     delayed(ParagraphBenchmark.compute_metric)(yt_region, yp_region, metric, 0.5)
                    #     for metric in metrics_to_compute
                    # )

                    for metric in metrics_to_compute:
                        ben_pdbs.append(str(pdb))
                        ben_chain_ids.append(str(chain))
                        ben_chain_types.append(str(chain_type))
                        ben_sequences.append(str(region_seq))
                        ben_imgts.append(ab_imgt_region)
                        ben_models.append(str(model_name))
                        ben_regions.append(str(region_name))
                        ben_scores.append(float(ParagraphBenchmark.compute_metric(yt_region, yp_region, metric, 0.5)))
                        ben_metrics.append(metric)
                        ben_predictions.append(yp_region)
                        ben_labels.append(yt_region)

            # Emptying memory to load new model
            # Coupled with the given device this provides a hacky way to keep models on the same GPU between iterations.
            del model
            if "cuda" in device:
                torch.cuda.empty_cache()

        ben_df = pl.DataFrame({
            "pdb": ben_pdbs,
            "antibody_chain": ben_chain_ids,
            "chain_type": ben_chain_types,
            "antibody_sequence": ben_sequences,
            "antibody_imgt": ben_imgts,
            "model": ben_models,
            "region": ben_regions,
            "value": ben_scores,
            "metric": ben_metrics,
            "predictions": ben_predictions,
            "full_paratope_labels": ben_labels,
        })
        ben_df.write_parquet(save_path)
        print(ben_df.dtypes)


if __name__ == '__main__':
    # Be cautious here, this is disabled only to not show the zero_division warning from sklearn when computing
    # metrics. Consider removing these two lines when developing!
    import warnings

    warnings.filterwarnings('ignore')

    print("Models currently accepted: ")
    [print(m) for m in ACCEPTABLE_MODELS]
    print()

    ParagraphBenchmark().run()

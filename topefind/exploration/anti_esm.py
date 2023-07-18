import random
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from tqdm import tqdm
from torch.utils.data import Dataset
from sklearn.ensemble import RandomForestClassifier

from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

from topefind.data_hub import SabdabHub
from topefind.embedders import ESMEmbedder, RemoteEmbedderName
from topefind.predictors import PLMSKClassifier
from topefind.utils import TOPEFIND_PATH, get_device

ESM_MODEL = RemoteEmbedderName.esm2_8m
LABELED_SABDAB = TOPEFIND_PATH / "datasets" / "SAbDab" / "sabdab.parquet"
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

SEED = 42
N_ESTIMATORS = 256
TEST_SIZE = 100
N_JOBS = 64

# Fix python random to seed
random.seed(42)


class AntiESMEmbedder(ESMEmbedder):
    def __init__(
            self,
            name: str = "anti_esm",
            device: str = "auto",
    ):
        self.name = name
        self.device = get_device(device)
        self.tokenizer = tokenizer.from_pretrained(ESM_MODEL)
        self.model = model.from_pretrained(name)
        self.model.to(self.device)
        self.model.eval()
        self.model.config.output_hidden_states = True


class AbAgDataset(Dataset):
    def __init__(self, sequences, tokenizer, block_size):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, i):
        return self.tokenizer(
            self.sequences[i],
            truncation=True,
            max_length=self.block_size,
            padding='max_length',
        )


# ----------------------------------------------------------------------------------------------------------------------
# GET DATA
# ----------------------------------------------------------------------------------------------------------------------
# Load the curated, labelled, compact dataset derived from SAbDab.
df: pd.DataFrame = pd.read_parquet(LABELED_SABDAB, columns=INTERESTED_COLUMNS)
# Let's filter according to literature guidelines.
df = df.drop_duplicates("antibody_sequence")  # Don't bias the model.
df = df[(df.antibody_sequence.str.len() > 70) & (df.antibody_sequence.str.len() < 200)]  # Don't go < 70 ...ANARCI.
df = df[df.full_paratope_labels.apply(sum) >= 1]  # At least some positives.
df = df[(df.num_antigen_chains > 0) & (df.num_antigen_chains <= 3)]  # Follows the choice in Paragraph.
df = df[~df.scfv]  # Hard to deal with since two chains are connected, kind of everyone drops them for now.
df = df[df.antigen_type.isin([
    "protein",
    "peptide",
    "protein | protein",
    "protein | peptide",
    "peptide | protein",
    "peptide | peptide",
])]
df = df[df.resolution < 3.5]  # Allows to define contacts above this resolution (used everywhere in literature).
df = df.reset_index()
# Now the set of unique ["antibody_sequence", "antibody_chain", "chain_type", "antigen_chain"]
# needs to be equal to the total number of elements in the DataFrame.
assert len(np.unique(["_".join(el) for el in df[PRIMARY_KEY].values])) == len(df), "Mismatch in primary key"
df = SabdabHub.fix_imgts(df)
# Done, a working dataset.
print(f"Dataset now contains {len(df)} entries")
print(f"{len(df[df.num_antigen_chains > 1])} entries are connected to multiple antigens")

all_df = df
df = df.sample(frac=1, replace=False, random_state=SEED)

# Take a holdout set that won't be seen by anything
ab_sequences = df["antibody_sequence"].to_list()
ag_sequences = df["antigen_sequence"].to_list()
ab_unseen = ab_sequences[-TEST_SIZE:]
ag_unseen = ag_sequences[-TEST_SIZE:]
ab_sequences = ab_sequences[:-TEST_SIZE]
ag_sequences = ag_sequences[:-TEST_SIZE]

labels_train = df["full_paratope_labels"].to_list()[:-TEST_SIZE]
labels_unseen = df["full_paratope_labels"].to_list()[-TEST_SIZE:]

# Use joint sequences and normal ones
ab_ag_seqs = [f"{ab}B{ag}" for ab, ag in zip(ab_sequences, ag_sequences)] + ab_sequences + ag_sequences
ab_ag_seqs = [seq for seq in ab_ag_seqs if len(seq) < 1024]
ab_ag_seqs_lens = [len(seq) for seq in ab_ag_seqs]

print("Paired Sequences Stats: ")
print(f"Max: {max(ab_ag_seqs_lens)}")
print(f"Mean: {np.mean(ab_ag_seqs_lens)}")
print(f"Min: {min(ab_ag_seqs_lens)}")
print(f"Len: {len(ab_ag_seqs)}")

random.shuffle(ab_ag_seqs)
lm_train_seqs = [seq for seq in ab_ag_seqs[:-TEST_SIZE]]
lm_val_seqs = [seq for seq in ab_ag_seqs[-TEST_SIZE:]]

# ----------------------------------------------------------------------------------------------------------------------
# FINE-TUNE ESM ON AB AG PAIRED SEQUENCES
# ----------------------------------------------------------------------------------------------------------------------
model = AutoModelForMaskedLM.from_pretrained(ESM_MODEL)
tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL)

train = AbAgDataset(lm_train_seqs, tokenizer, block_size=128)
val = AbAgDataset(lm_val_seqs, tokenizer, block_size=128)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

training_args = TrainingArguments(
    output_dir='./results',
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    logging_dir='./logs',
    logging_steps=30,
    learning_rate=5e-5,
    do_train=True,
    do_eval=True,
    evaluation_strategy="steps",
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train,
    eval_dataset=val,
)

trainer.train()
trainer.save_model("anti_esm")

# ----------------------------------------------------------------------------------------------------------------------
# PREDICT FROM EMBEDDINGS BY TRAINING A CLASSIFIER
# ----------------------------------------------------------------------------------------------------------------------
# Define models
anti_esm = AntiESMEmbedder("anti_esm")
rf = RandomForestClassifier(n_estimators=N_ESTIMATORS, n_jobs=N_JOBS, random_state=SEED)

# Merge them
anti_esm_rf = PLMSKClassifier(anti_esm, rf)

# Train the head
anti_esm_rf.train(ab_sequences, labels_train)

# Check train AP
preds = [anti_esm_rf.predict(seq) for seq in tqdm(ab_sequences)]
aps = [average_precision_score(yt, yp) for yt, yp in zip(labels_train, preds)]
print(f"Train AP: {np.mean(aps)} +/- {np.std(aps)}")
# Check test AP
preds = [anti_esm_rf.predict(seq) for seq in tqdm(ab_unseen)]
aps = [average_precision_score(yt, yp) for yt, yp in zip(labels_unseen, preds)]
print(f"Test AP: {np.mean(aps)} +/- {np.std(aps)}")

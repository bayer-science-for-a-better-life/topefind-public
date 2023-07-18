import click
import torch
import os
import json
import logging
import sys
from pprint import pprint
from typing import Optional, Tuple

from parapred.model import Parapred, clean_output
from parapred.cnn import generate_mask
from parapred.preprocessing import encode_batch

MAX_PARAPRED_LEN = 40

LOGGER = logging.getLogger("Parapred-Logger")
LOGGER.setLevel(logging.INFO)

WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "parapred/weights/parapred_pytorch.h5")


@click.command(help = "Predict paratope probability for a single CDR sequence.")
@click.argument("cdr")
@click.option("--weight", "-w", help = "Specify path for weights.")
@click.option("--output", "-o", help = "Specify output JSON filename for prediction.", default = "output.json")
@click.option("--no-output", "-no", help = "Do not write an output file.", default = False, is_flag=True)
@click.option("--verbose", "-v", help = "Be verbose.", default = False, is_flag=True)
@click.option("--sigmoid", "-s", help = "Use sigmoid activation.", default = False, is_flag=True)
def predict(cdr: str,
            weight: Optional[str] = None,
            output: str = "output.json",
            no_output: bool = False,
            verbose: bool = False,
            sigmoid: bool = False):

    if len(cdr) > MAX_PARAPRED_LEN:
        LOGGER.error(f"Length of the CDR sequence ({len(cdr)}) is too long. Unsupported.")
        sys.exit(1)
    elif len(cdr) < 4:
        LOGGER.error(f"The original Parapred method requires at least 2 amino acids flanking the CDR.")
        sys.exit(1)

    # Encode input sequences
    if verbose:
        LOGGER.info(f"Encoding CDR sequence {cdr}")

    sequences, lengths = encode_batch([cdr], max_length=MAX_PARAPRED_LEN)

    # Generate a mask for the input
    m = generate_mask(sequences, sequence_lengths=lengths)

    # load pre-trained parapred model
    activation = "sigmoid" if sigmoid else "hard_sigmoid"
    if verbose and sigmoid:
        LOGGER.info(f"Using sigmoid activation in the LSTM.")

    p = Parapred(lstm_activation=activation)

    # load weights
    if weight is not None:
        if verbose:
            LOGGER.info(f"Loading weights from {weight}")
        try:
            p.load_state_dict(torch.load(weight))
        except IOError:
            LOGGER.warning(f"Pre-trained weights file {weight} cannot be detected. Defaulting to pre-trained weights.")
            p.load_state_dict(torch.load(weight))

    elif weight is None:
        if verbose:
            LOGGER.info(f"Loading pre-trained weights.")
        p.load_state_dict(torch.load(WEIGHTS_PATH))

    # Evaluation mode with no gradient computations
    _ = p.eval()
    with torch.no_grad():
        probabilities = p(sequences, m, lengths)

    # Linearise probabilities for viewing
    out = {}
    clean = clean_output(probabilities, lengths[0]).tolist()

    i_prob = [round(_, 5) for i, _ in enumerate(clean)]
    seq_to_prob = list(zip(cdr, i_prob))
    out[cdr] = seq_to_prob

    if verbose:
        pprint(out)

    if no_output is False:
        if verbose:
            LOGGER.info(f"Writing results to {output}")
        with open(output, "w") as jason:
            json.dump(out, jason)


@click.command(help = "Predict if two CDR sequences belong to the same paratype.")
@click.argument("cdrs", nargs=2)
@click.option("--weight", "-w", help = "Specify path for weights.")

@click.option("--sigmoid", "-s", help = "Use sigmoid activation.", default = False, is_flag=True)
@click.option("--threshold", "-t", help = "Specify paratope probability threshold. Defualt: 0.67", default = 0.67)
def paratype(cdrs: Tuple[str, str],
            weight: Optional[str] = None,
            sigmoid: bool = False,
            threshold: float = 0.67):

    # do some sanity checks
    cdr_a, cdr_b = cdrs
    if len(cdr_a) != len(cdr_b):
        LOGGER.error("CDRs do not have identical length.")
        sys.exit(1)

    if len(cdr_a) > MAX_PARAPRED_LEN:
        LOGGER.error(f"Length of the CDR sequences ({len(cdr_a)}) is too long. Unsupported.")
        sys.exit(1)

    elif len(cdr_a) <= 4:
        LOGGER.error(f"Parapred requires at least 2 amino acids flanking the CDR. Too short!")
        sys.exit(1)

    # Encode input sequences
    sequences, lengths = encode_batch(cdrs, max_length=MAX_PARAPRED_LEN)

    # Generate a mask for the input
    m = generate_mask(sequences, sequence_lengths=lengths)

    # load pre-trained parapred model
    activation = "sigmoid" if sigmoid else "hard_sigmoid"
    if sigmoid:
        LOGGER.info(f"Using sigmoid activation in the LSTM.")

    p = Parapred(lstm_activation=activation)

    # load weights
    if weight is not None:
        try:
            p.load_state_dict(torch.load(weight))
        except IOError:
            LOGGER.warning(f"Pre-trained weights file {weight} cannot be detected. Defaulting to pre-trained weights.")
            p.load_state_dict(torch.load(weight))

    elif weight is None:
        p.load_state_dict(torch.load(WEIGHTS_PATH))

    # Evaluation mode with no gradient computations
    _ = p.eval()
    with torch.no_grad():
        probabilities = p(sequences, m, lengths)

    # Linearise probabilities for viewing
    probabilities = [clean_output(pr, lengths[i].item()) for i, pr in enumerate(probabilities)]
    seq_to_prob = [list(zip(cdrs[i], probabilities[i].detach().numpy())) for i, pr in enumerate(probabilities)]

    paratope_a_total, paratope_b_total = 0, 0
    matches = 0
    aln = ""

    # Iterate through position-by-position
    for pos_a, pos_b in zip(seq_to_prob[0], seq_to_prob[1]):
        a_is_paratope = pos_a[1] >= threshold
        b_is_paratope = pos_b[1] >= threshold

        paratope_a_total += (a_is_paratope)
        paratope_b_total += (b_is_paratope)

        if a_is_paratope and b_is_paratope and (pos_a[0] == pos_b[0]):
            matches += 1
            aln += ":"
        elif a_is_paratope and b_is_paratope and (pos_a[0] != pos_b[0]):
            aln += "x"
        elif a_is_paratope and not b_is_paratope:
            aln += "'"
        elif b_is_paratope and not a_is_paratope:
            aln += "."
        else:
            aln += " "

    if min(paratope_a_total, paratope_b_total) == 0:
        print(f"{cdr_a}\n{aln}\n{cdr_b}\nParatype Score=0.00")
    else:
        paratype_score = matches / min(paratope_a_total, paratope_b_total)
        print(f"{cdr_a}\n{aln}\n{cdr_b}\nParatype Score={paratype_score:.3f}")


@click.group()
def cli():
    pass

cli.add_command(predict)
cli.add_command(paratype)

if __name__ == "__main__":
    cli()

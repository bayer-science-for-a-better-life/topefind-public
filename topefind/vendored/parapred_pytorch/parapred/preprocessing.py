from topefind.vendored.parapred_pytorch.parapred.utils import *
from typing import List

NUM_FEATURES = NUM_AMINOS + NUM_MEILER


def encode_parapred(sequence: str, max_length: int = None) -> torch.Tensor:
    """
    One-hot encode an amino acid sequence, then concatenate with Meiler features.

    :param sequence:   CDR sequence
    :param max_length: specify the maximum length for a CDR sequence

    :return: max_length x num_features tensor
    """
    # First one-hot encode the sequence, then fill the rest as the meiler feature for that amino acid
    seqlen = len(sequence)
    if max_length is None:
        encoded = torch.zeros((seqlen, NUM_FEATURES))
    else:
        encoded = torch.zeros((max_length, NUM_AMINOS + NUM_MEILER))

    for i, c in enumerate(sequence):
        encoded[i][PARAPRED_TO_POS.get(c, NUM_AMINOS)] = 1
        encoded[i][-NUM_MEILER:] = MEILER[c]

    return encoded


def encode_batch(batch_of_sequences: List[str], max_length: int = None):
    """
    Encode a batch of sequences into tensors, along with their lengths

    :param batch_of_sequences:
    :param max_length:
    :return:
    """
    encoded_seqs = []
    seq_lens = []

    for seq in batch_of_sequences:
        encoded_seqs.append(encode_parapred(seq, max_length=max_length))
        seq_lens.append(len(seq))

    # Convert list of Tensors into a bigger tensor
    try:
        encoded_seqs = torch.stack(encoded_seqs)
    except RuntimeError:
        # The expectation is that torch.stack should allow us to create a bigger tensor
        # if not, we should pad the sequences.
        encoded_seqs = torch.nn.utils.rnn.pad_sequence(encoded_seqs, batch_first=True)

    # Parapred first applies a CNN to an input tensor.
    # CNNs in PyTorch expect a tensor T of (Bsz x n_features x seqlen)
    # Hence the permutation
    encoded_seqs = encoded_seqs.permute(0, 2, 1)

    return encoded_seqs, torch.as_tensor(seq_lens)

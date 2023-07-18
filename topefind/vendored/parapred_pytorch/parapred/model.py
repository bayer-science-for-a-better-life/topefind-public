import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from topefind.vendored.parapred_pytorch.parapred.cnn import Masked1dConvolution
from topefind.vendored.parapred_pytorch.parapred.hslstm import LSTMHardSigmoid

# accepts CDRs up to length 28 + 2 residues either side
PARAPRED_MAX_LEN = 40

# 21 amino acids + 7 meiler features
PARAPRED_N_FEATURES = 28

# kernel size as per Parapred
PARAPRED_KERNEL_SIZE = 3


class Parapred(nn.Module):
    """
    Main Parapred model

    Details of model architecture in Liberis et al., 2018
    https://academic.oup.com/bioinformatics/article/34/17/2944/4972995

    """
    def __init__(self,
                 input_dim: int = PARAPRED_MAX_LEN,
                 output_dim: int = PARAPRED_MAX_LEN,
                 n_channels: int = PARAPRED_N_FEATURES,
                 kernel_size: int = PARAPRED_KERNEL_SIZE,
                 n_hidden_cells: int = 256,
                 lstm_activation: str = "hard_sigmoid"):
        """

        :param input_dim:
        :param output_dim:
        :param n_channels:
        :param kernel_size:
        :param n_hidden_cells:
        :param lstm_activation:
        """

        super().__init__()
        self.mconv = Masked1dConvolution(
            input_dim,
            in_channels=n_channels,
            output_dim=output_dim,
            out_channels=n_channels,
            kernel_size=kernel_size
        )
        self.elu = nn.ELU()

        # Keeping batch first as that's how it comes from the CNN
        # We offer two activation options; the hard sigmoid which
        # was the Keras default at the time of Parapred's publication,
        # or the regular sigmoid function which is now the standard
        # for both PyTorch and Keras/TF.
        if lstm_activation == 'sigmoid':
            self.lstm = nn.LSTM(
                input_size=n_channels,
                hidden_size=n_hidden_cells,
                batch_first=True,
                bidirectional=True,
            )
        elif lstm_activation == 'hard_sigmoid':
            self.lstm = LSTMHardSigmoid(
                input_size=n_channels,
                hidden_size=n_hidden_cells,
                batch_first=True,
                bidirectional=True,
            )

        # need to multiply by 2 as it's a bidirectional LSTM.
        self.fc = nn.Linear(n_hidden_cells*2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor: torch.Tensor, mask: torch.BoolTensor, lengths: torch.LongTensor) -> torch.Tensor:
        """
        Forward pass of Parapred given the input, mask, and sequence lengths

        :param input_tensor: an input tensor of (bsz x features x seqlen)
        :param mask: a boolean tensor of (bsz x 1 x seqlen)
        :param lengths: a LongTensor of (seqlen); must be equal to bsz
        :return:
        """
        # residual connection following ELU
        o = input_tensor + self.elu(self.mconv(input_tensor, mask))

        # Packing sequences to remove padding
        packed_seq = pack_padded_sequence(o.permute(0, 2, 1), lengths, batch_first=True, enforce_sorted=True)
        o_packed, (h, c) = self.lstm(packed_seq)

        # Re-pad sequences before prediction of probabilities
        o, lengths = pad_packed_sequence(o_packed, batch_first=True, total_length=PARAPRED_MAX_LEN)

        # Predict probabilities
        o = self.sigmoid(self.fc(o))

        return o


def clean_output(output_tensor: torch.Tensor, sequence_length: int) -> torch.Tensor:
    """
    Clean the output tensor of probabilities to remove the predictions for padded positions

    :param output_tensor: output from the Parapred model; shape: (max_length x 1)
    :param sequence_length: length of sequence

    :return:
    """
    return output_tensor[:sequence_length].view(-1)

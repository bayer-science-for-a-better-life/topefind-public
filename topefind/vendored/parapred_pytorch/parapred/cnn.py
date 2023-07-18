import torch
import torch.nn as nn
from typing import Optional


class Masked1dConvolution(nn.Module):
    def __init__(self,
                 input_dim: int,
                 in_channels: int,
                 output_dim: Optional[int] = None,
                 out_channels: Optional[int] = None,
                 kernel_size: int = 3,
                 dilation: int = 1,
                 stride: int = 1,
                 ):
        """
        A "masked" 1d convolutional neural network.

        For an input tensor T (bsz x features x seqlen), apply a boolean mask M (bsz x features x seqlen) follwing
        convolution. This essentially "zeros out" some of the values following convolution.

        """
        super().__init__()

        # Assert same shape
        self.input_dim = input_dim
        self.output_dim = input_dim if output_dim is None else output_dim

        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels is None else out_channels

        # Determine the padding required for keeping the same sequence length
        assert dilation >= 1 and stride >= 1, "Dilation and stride must be >= 1."
        self.dilation, self.stride = dilation, stride
        self.kernel_size = kernel_size

        padding = self.determine_padding(self.input_dim, self.output_dim)

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            self.kernel_size,
            padding=padding)

    def forward(self, x: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        """
        Forward pass.

        :param x: an input tensor of (bsz x in_channels x seqlen)
        :param mask: a mask tensor (boolean) of (bsz x out_channels x seqlen)
        :return:
        """
        assert x.shape == mask.shape, \
            f"Shape of input tensor ({x.size()[0]}, {x.size()[1]}, {x.size()[2]}) " \
            f"does not match mask shape ({mask.size()[0]}, {mask.size()[1]}, {mask.size()[2]})."

        # Run through a regular convolution
        o = self.conv(x)

        # Apply the mask to "zero out" positions beyond sequence length
        return o * mask

    def determine_padding(self, input_shape: int, output_shape: int) -> int:
        """
        Determine the padding required to keep the same length (i.e. padding='same' from Keras)
        https://pytorch.org/docs/master/generated/torch.nn.Conv1d.html

        L_out = ((L_in + 2 x padding - dilation x (kernel_size - 1) - 1)/stride + 1)

        :return: An integer defining the amount of padding required to keep the "same" padding effect
        """
        padding = (((output_shape - 1) * self.stride) + 1 - input_shape + (self.dilation * (self.kernel_size - 1)))

        # integer division
        padding = padding // 2
        assert output_shape == l_out(
            input_shape, padding, self.dilation, self.kernel_size, self.stride
        ) and padding >= 0, f"Input and output of {input_shape} and {output_shape} with " \
           f"kernel {self.kernel_size}, dilation {self.dilation}, stride {self.stride} " \
           f"are incompatible for a Conv1D network."
        return padding


def generate_mask(input_tensor: torch.Tensor, sequence_lengths: torch.LongTensor) -> torch.Tensor:
    """
    Generate a mask for masked 1d convolution.

    :param input_tensor: an input tensor for convolution (bsz x features x seqlen)
    :param sequence_lengths: length of sequences (bsz,)
    :return:
    """
    assert input_tensor.size()[0] == sequence_lengths.size()[0], \
        f"Batch size {input_tensor.size()[0]} != number of provided lengths {sequence_lengths.size()[0]}."

    mask = torch.ones_like(input_tensor, dtype = torch.bool)
    for i, length in enumerate(sequence_lengths):
        mask[i][:, length:] = False

    return mask


def l_out(l_in: int, padding: int, dilation: int, kernel: int, stride: int) -> int:
    """
    Determine the L_out of a 1d-CNN model given parameters for the 1D CNN

    :param l_in: length of input
    :param padding: number of units to pad
    :param dilation: dilation for CNN
    :param kernel: kernel size for CNN
    :param stride: stride size for CNN
    :return:
    """
    return (l_in + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1

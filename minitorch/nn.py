from typing import Tuple, Optional
import builtins
from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand, tensor


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    nheight = height // kh
    nwidth = width // kw

    # make sure everything is contiguous before doing any view magic
    input = input.contiguous()

    # split row by kw and split column by kh
    input = input.view(batch, channel, nheight, kh, nwidth, kw)

    # reorg so kernels are together in order to flatten
    input = input.permute(0, 1, 2, 4, 3, 5)

    # flatten kernel
    input = input.contiguous().view(batch, channel, nheight, nwidth, kh * kw)

    return input, nheight, nwidth


def avgpool2d(t1: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Averages every kernel mapping in the Tensor"""
    batch, channel = t1.shape[0], t1.shape[1]
    tiled, tile_height, tile_width = tile(t1, kernel)
    return tiled.mean(dim=len(tiled.shape) - 1).view(
        batch, channel, tile_height, tile_width
    )


max_reduce = FastOps.reduce(operators.max, float("-inf"))


def argmax(t1: Tensor, dim: Optional[Tensor] = None) -> Tensor:
    """Creates a one(possibly multi for duplciates) hot encoding on the given dimension of the max value"""
    if dim is not None:
        reduced = max_reduce(t1, int(dim.item()))
        res = reduced == t1
        return res
    else:
        flat = t1.contiguous().view(t1.size)
        total_max = builtins.max(flat._tensor._storage)
        res = flat == total_max
        return res.view(*t1.shape)


class Max(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, dim: Tensor) -> Tensor:
        """Forward pass of the Max function"""
        ctx.save_for_backward(t1, dim)
        return max_reduce(t1, int(dim.item()))

    @staticmethod
    def backward(ctx: Context, grad: Tensor) -> Tuple[Tensor, Tensor]:
        """Backward pass of the Max function"""
        t1, dim = ctx.saved_values
        grad_output = argmax(t1, dim) * grad
        grad_output = grad_output.sum(dim=int(dim.item()))
        return grad_output, tensor([0.0])


def max(t1: Tensor, dim: Optional[int]) -> Tensor:
    """Applies the max function to a tensor"""
    if dim is not None:
        return Max.apply(t1, t1._ensure_tensor(dim))
    else:
        # flatten first since we will just max everything
        return Max.apply(t1.contiguous().view(t1.size), tensor([0]))


def maxpool2d(t1: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Performs a 2D max pooling given a kernel over a tensor"""
    batch, channel = t1.shape[0], t1.shape[1]
    tiled, tile_height, tile_width = tile(t1, kernel)
    return max_reduce(tiled, len(tiled.shape) - 1).view(
        batch, channel, tile_height, tile_width
    )


def softmax(t1: Tensor, dim: Optional[int]) -> Tensor:
    """Replaces a dimension with a softmax of that entire dimension"""
    if dim is None:
        # flatten
        t1 = t1.contiguous().view(t1.size)
        dim = 0
    exponentiated = t1.exp()
    new_shape = list(exponentiated.shape)
    new_shape[dim] = 1
    return exponentiated / exponentiated.sum(dim=dim).contiguous().view(*new_shape)


def dropout(t1: Tensor, prob: float, ignore: bool = False) -> Tensor:
    """Randomly samples a tensor based on a probability and either drops or keeps values"""
    if ignore or prob <= 0:
        return t1
    elif prob >= 1:
        return t1.zeros()
    return t1 * (rand(t1.shape) > prob) / (1 - prob)


def logsoftmax(t1: Tensor, dim: Optional[int]) -> Tensor:
    """Performs the logsoftmax on the tensor on the given dimension"""
    if dim is None:
        t1 = t1.contiguous().view(t1.size)
        dim = 0
    maxv = max(t1, dim)
    logsumexp = (t1 - maxv).exp().sum(dim=dim).log() + maxv
    return t1 - logsumexp


from torch._six import container_abcs
from itertools import repeat, product


def _triple(x):
    """If x is a single number, repeat three times."""
    if isinstance(x, container_abcs.Iterable):
        return x
    return tuple(repeat(x, 3))


# copy form pytorch
def _calculate_fan_in_and_fan_out_hwio(tensor):
    """Init convolution weight. Copied from pytorch."""
    dimensions = tensor.ndimension()
    if dimensions < 2:
        raise ValueError(
            "Fan in and fan out can not be computed for tensor with fewer than 2 dimensions"
        )

    if dimensions == 2:  # Linear
        fan_in = tensor.size(-2)
        fan_out = tensor.size(-1)
    else:
        num_input_fmaps = tensor.size(-2)
        num_output_fmaps = tensor.size(-1)
        receptive_field_size = 1
        if tensor.dim() > 2:
            receptive_field_size = tensor[..., 0, 0].numel()
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out
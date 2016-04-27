import math
import logging

from .base import *


logger = logging.getLogger(__name__)


def make_shape(n, c, h, w):
    logger.debug("Output shape of the layer is: %s", (n, c, h, w))
    return n, c, h, w


def get_filter_output_shape(i_h, i_w, params, round_func):    
    o_h = (i_h + 2 * params.pad_h - params.kernel_h)/float(params.stride_h) + 1
    o_w = (i_w + 2 * params.pad_w - params.kernel_w)/float(params.stride_w) + 1
    return int(round_func(o_h)), int(round_func(o_w))


def get_reverse_filter_output_shape(i_h, i_w, params):
    """
    Compute output shape for the reverse filter. Computation is based on:
    https://github.com/longjon/caffe/blob/25c2e3fdc7a27ec00893bd0335ac42e53ff2c7aa/src/caffe/layers/deconv_layer.cpp#L12

    :param i_h: height of the input
    :param i_w: width of the input
    :param params: namedtuple KernelParameters

    :return: output_height, output_width
    """
    o_h = ((i_h - 1) * params.stride_h) + params.kernel_h - 2 * params.pad_h
    o_w = ((i_w - 1) * params.stride_w) + params.kernel_w - 2 * params.pad_w
    return o_h, o_w


def get_strided_kernel_output_shape(node, round_func, deconvolution=False):
    assert node.layer is not None
    logger.info("computing shape for %s layer", node.kind)
    input_shape = node.get_only_parent().output_shape
    if deconvolution:
        o_h, o_w = get_reverse_filter_output_shape(input_shape[IDX_H],
                                                   input_shape[IDX_W],
                                                   node.layer.kernel_parameters)
    else:
        o_h, o_w = get_filter_output_shape(input_shape[IDX_H],
                                           input_shape[IDX_W],
                                           node.layer.kernel_parameters,
                                           round_func)
    params = node.layer.parameters
    has_c_o = hasattr(params, 'num_output')
    c = params.num_output if has_c_o else input_shape[IDX_C]
    return make_shape(input_shape[IDX_N], c, o_h, o_w)


def shape_not_implemented(node):
    """
    In case that shape is not implemented

    :param node: node, it's necessary, because of the kaffe/layers.py
    """
    raise NotImplementedError


def shape_identity(node):
    assert len(node.parents) > 0
    return node.parents[0].output_shape


def shape_scalar(node):
    """
    Shape scalar

    :param node: node, it's necessary, because of the kaffe/layers.py
    """
    return make_shape(1, 1, 1, 1)


def shape_data(node):
    if node.output_shape:
        # Old-style input specification
        return node.output_shape
    try:
        # New-style input specification
        return map(int, node.parameters.shape[0].dim)
    except:
        pass
    # We most likely have a data layer on our hands. The problem is,
    # Caffe infers the dimensions of the data from the source (eg: LMDB).
    # We want to avoid reading datasets here. Fail for now.
    # This can be temporarily fixed by transforming the data layer to
    # Caffe's "input" layer (as is usually used in the "deploy" version).
    # TODO: Find a better solution for this.
    raise KaffeError('Cannot determine dimensions of data layer.\n'
                     'See comments in function shape_data for more info.')


def shape_mem_data(node):
    params = node.parameters
    return make_shape(params.batch_size,
                      params.channels,
                      params.height,
                      params.width)


def shape_concat(node):
    axis = node.layer.parameters.axis
    output_shape = None
    for parent in node.parents:
        if output_shape is None:
            output_shape = list(parent.output_shape)
        else:
            output_shape[axis] += parent.output_shape[axis]
    return tuple(output_shape)


def shape_convolution(node):
    return get_strided_kernel_output_shape(node, math.floor)


def shape_deconvolution(node):
    """
    Compute shape for the deconvolution layer.

    :param node: class Node representing the deconvolution layer

    :return: output shape of the deconvolution layer
    """
    return get_strided_kernel_output_shape(node, math.ceil, deconvolution=True)


def shape_crop(node):
    """
    Compute shape for the crop layer.

    :param node: class Node representing the crop layer

    :return: output shape of the crop layer
    """
    n, c, h, w = node.parents[1].output_shape
    return make_shape(n, c, h, w)


def shape_pool(node):
    return get_strided_kernel_output_shape(node, math.ceil)


def shape_inner_product(node):
    input_shape = node.get_only_parent().output_shape
    return make_shape(input_shape[IDX_N],
                      node.layer.parameters.num_output,
                      1, 1)

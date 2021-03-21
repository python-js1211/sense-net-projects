"""These functions are used to generate tensorflow layers from a
topology spcification given in a dictionary, and to extract learned
parameters from the layers to export the model.

"""

import sensenet.importers
tf = sensenet.importers.import_tensorflow()
kl = sensenet.importers.import_keras_layers()

from sensenet.layers.utils import activation_function, build_graph
from sensenet.layers.core import CORE_LAYERS
from sensenet.layers.convolutional import CONVOLUTIONAL_LAYERS

SIMPLE_LAYERS = {}
SIMPLE_LAYERS.update(CORE_LAYERS)
SIMPLE_LAYERS.update(CONVOLUTIONAL_LAYERS)

class BlockMaker():
    def __init__(self, params, path_names, block_type):
        self._paths = [params[name] for name in path_names]
        self._activator = kl.Activation(activation_function(params))

    def __call__(self, inputs):
        outputs = []

        for path in self._paths:
            if len(path) > 0:
                path_outs = build_graph(path, SIMPLE_LAYERS, inputs)
                outputs.append(path_outs[-1].output)
            else:
                outputs.append(inputs)

        activations = kl.Add()(outputs)
        return self._activator(activations)

def dense_res_block(params):
    paths = ['dense_path', 'identity_path']
    type_str = 'dense_residual_block'

    return BlockMaker(params, paths, type_str)

def xception_block(params):
    paths = ['separable_convolution_path', 'single_convolution_path']
    type_str = 'xception_block'

    return BlockMaker(params, paths, type_str)

def resnet_block(params):
    paths = ['convolution_path', 'identity_path']
    type_str = 'resnet_block'

    return BlockMaker(params, paths, type_str)

def darknet_block(params):
    paths = ['convolution_path', 'identity_path']
    type_str = 'darknet_residual_block'

    return BlockMaker(params, paths, type_str)

def mobilev2_block(params):
    paths = ['convolution_path', 'identity_path']
    type_str = 'mobilenet_residual_block'

    return BlockMaker(params, paths, type_str)

BLOCKS = {
    'xception_block': xception_block,
    'dense_residual_block': dense_res_block,
    'resnet_block': resnet_block,
    'resnet18_block': resnet_block,
    'darknet_residual_block': darknet_block,
    'mobilenet_residual_block': mobilev2_block
}

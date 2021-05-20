import sensenet.importers

tf = sensenet.importers.import_tensorflow()

from sensenet.layers.block import SIMPLE_LAYERS, BLOCKS
from sensenet.layers.convolutional import CONVOLUTIONAL_LAYERS, get_shape_params
from sensenet.layers.legacy import build_legacy_graph
from sensenet.layers.tree import ForestPreprocessor
from sensenet.layers.utils import WEIGHT_INITIALIZERS, build_graph

LAYER_FUNCTIONS = {}
LAYER_FUNCTIONS.update(SIMPLE_LAYERS)
LAYER_FUNCTIONS.update(BLOCKS)

WEIGHTED_LAYERS = set()
WEIGHTED_LAYERS.update((CONVOLUTIONAL_LAYERS.keys()))
WEIGHTED_LAYERS.update(["dense", "batch_normalization"])


def feed_through(layers, inputs):
    if any(layer.get("type", None) is None for layer in layers):
        graph = build_legacy_graph(layers, inputs)
    else:
        try:
            graph = build_graph(layers, LAYER_FUNCTIONS, inputs)
        except:
            raise ValueError(remove_weights(layers))

    return graph[-1].output


def tree_preprocessor(model):
    if model.get("trees", None):
        return ForestPreprocessor(trees=model["trees"])
    else:
        return None


def get_n_nodes(params):
    if isinstance(params["weights"], str):
        return int(params["number_of_nodes"])
    elif "stdev" in params:
        return len(params["weights"])
    else:
        return len(params["weights"][0])


def remove_weights(element):
    if isinstance(element, dict):
        edict = dict(element)
        ltype = edict.get("type", None)

        if ltype in WEIGHTED_LAYERS or "weights" in edict:
            if ltype == "dense" or "weights" in edict:
                edict["number_of_nodes"] = get_n_nodes(edict)
            elif ltype in CONVOLUTIONAL_LAYERS:
                nfilters, kdims = get_shape_params(edict)
                edict["number_of_filters"] = nfilters
                edict["kernel_dimensions"] = kdims

            for key in edict:
                if key in WEIGHT_INITIALIZERS and isinstance(edict[key], list):
                    edict[key] = WEIGHT_INITIALIZERS[key]

        for key in edict:
            if isinstance(edict[key], (dict, list, tuple)):
                edict[key] = remove_weights(edict[key])

        return edict
    elif isinstance(element, (list, tuple)):
        return [remove_weights(e) for e in element]
    else:
        return element

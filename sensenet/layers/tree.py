"""Features generated via tree embedding.

This module is a wrapper for BigML deepnet "tree embeddings", which
is essentially a tree-based convolutional layer applied before the
fully-connected layers in a BigML deepnet. Note that here we define
both a custom layer and a custom operator, the latter of which relies
on a tensorflow extension (BigMLTreeify) which is compiled and
installed when sensenet is installed.

"""
import collections

import sensenet.importers
tf = sensenet.importers.import_tensorflow()
np = sensenet.importers.import_numpy()
tl = sensenet.importers.import_bigml_treelib()

from sensenet.layers.utils import constant

def into_arrays(node, all_nodes, outputs):
    if len(node) == 2 and node[-1] is None:
        all_nodes.append([-1, -1, len(outputs), len(outputs)])
        outputs.append(list(node[0]))
    else:
        this_node = [node[0], node[1], 0, 0]
        all_nodes.append(this_node)

        this_node[2] = len(all_nodes)
        into_arrays(node[2], all_nodes, outputs)

        this_node[3] = len(all_nodes)
        into_arrays(node[3], all_nodes, outputs)

def tree_to_arrays(atree):
    node_list = []
    outputs = []

    into_arrays(atree, node_list, outputs)

    return {
        'split_indices': np.array([n[0] for n in node_list], dtype=np.int32),
        'split_values': np.array([n[1] for n in node_list], dtype=np.float32),
        'left': np.array([n[2] for n in node_list], dtype=np.int32),
        'right': np.array([n[3] for n in node_list], dtype=np.int32),
        'outputs': np.array(outputs, dtype=np.float32)
    }

def trees_to_arrays(trees):
    tarrays = [tree_to_arrays(t) for t in trees]
    keys = list(tarrays[0].keys())
    maxes = {k: np.max([t[k].shape[0] for t in tarrays]) for k in keys}

    output_arrays = {}
    ntrees = len(tarrays)

    for k in keys:
        if k != 'outputs':
            atype = tarrays[0][k].dtype
            all_array = np.zeros((ntrees, maxes[k]), dtype=atype) - 1

            for i, tarraymap in enumerate(tarrays):
                tarraylen = tarraymap[k].shape[0]
                all_array[i, 0:tarraylen] = tarraymap[k]

            output_arrays[k] = all_array

    noutputs = tarrays[0]['outputs'].shape[1]
    prob_array = np.zeros((ntrees, maxes['outputs'], noutputs), dtype=np.float32)

    for i, tarraymap in enumerate(tarrays):
        problen = tarraymap['outputs'].shape[0]
        prob_array[i, 0:problen, :] = tarraymap['outputs']

    output_arrays['outputs'] = prob_array

    return output_arrays

class DecisionForest():
    def __init__(self, trees):
        self._trees = trees
        self._arrays = trees_to_arrays(trees)
        self._noutputs = self._arrays['outputs'].shape[-1]

    def __call__(self, inputs):
        return tl.BigMLTreeify(points=inputs,
                               split_indices=self._arrays['split_indices'],
                               split_values=self._arrays['split_values'],
                               left=self._arrays['left'],
                               right=self._arrays['right'],
                               outputs=self._arrays['outputs'])

class ForestPreprocessor():
    def __init__(self, trees=None):
        self._trees = trees
        self._forests = []
        self._ranges = []

        if trees:
            for input_range, trees in self._trees:
                self._forests.append([input_range, DecisionForest(trees)])

    def __call__(self, inputs):
        all_preds = []

        for input_range, forest in self._forests:
            start, end = input_range
            tree_inputs = inputs[:,start:end]
            all_preds.append(forest(tree_inputs))

        return tf.concat(all_preds + [inputs], -1)

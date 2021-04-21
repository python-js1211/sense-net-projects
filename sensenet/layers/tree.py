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

class DecisionNode():
    def __init__(self, **kwargs):
        self._tree = kwargs['tree']
        assert len(self._tree) == 2 and self._tree[-1] is None
        self._outputs = self._tree[0]

    def __call__(self, inputs):
        output_tensor = tf.reshape(constant(self._outputs), [1, -1])
        return tf.tile(output_tensor, [tf.shape(inputs)[0], 1])

class DecisionTree():
    def __init__(self, **kwargs):
        self._tree = kwargs['tree']

        node_list = []
        outputs = []

        into_arrays(self._tree, node_list, outputs)

        self._split_indices = np.array([n[0] for n in node_list], dtype=np.int32)
        self._split_values = np.array([n[1] for n in node_list], dtype=np.float32)
        self._left = np.array([n[2] for n in node_list], dtype=np.int32)
        self._right = np.array([n[3] for n in node_list], dtype=np.int32)
        self._outputs = np.array(outputs, dtype=np.float32)

    def __call__(self, inputs):
        out_idxs = tl.BigMLTreeify(points=inputs,
                                   split_indices=self._split_indices,
                                   split_values=self._split_values,
                                   left=self._left,
                                   right=self._right)

        return tf.gather(self._outputs, tf.reshape(out_idxs, (-1,)))

class DecisionForest():
    def __init__(self, **kwargs):
        self._forest = kwargs['forest']
        self._trees = []

        for tree in self._forest:
            if len(tree) == 2 and tree[-1] is None:
                self._trees.append(DecisionNode(tree=tree))
            else:
                self._trees.append(DecisionTree(tree=tree))

    def __call__(self, inputs):
        all_preds = []

        for tree in self._trees:
            preds = tree(inputs)
            all_preds.append(preds)

        summed = tf.math.add_n(all_preds)

        return summed / len(all_preds)

class ForestPreprocessor(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        newargs = dict(kwargs)
        self._trees = newargs.pop('trees')
        super().__init__(**newargs)

        self._forests = []
        self._ranges = []

        for input_range, trees in self._trees:
            self._forests.append([input_range, DecisionForest(forest=trees)])

    def call(self, inputs):
        all_preds = []

        for input_range, forest in self._forests:
            start, end = input_range
            tree_inputs = inputs[:,start:end]
            all_preds.append(forest(tree_inputs))

        return tf.concat(all_preds + [inputs], -1)

    def get_config(self):
        config = super().get_config()
        config['trees'] = self._trees

        return config

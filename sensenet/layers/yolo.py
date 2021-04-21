import sensenet.importers
tf = sensenet.importers.import_tensorflow()

from sensenet.accessors import get_image_shape, yolo_outputs
from sensenet.layers.construct import LAYER_FUNCTIONS
from sensenet.layers.utils import build_graph

def yolo_decode(features, decoding_info, input_size, nclasses):
    strides, anchors, xyscale = decoding_info
    fsize = tf.shape(features)[0]
    osize = tf.constant(input_size // int(strides))
    ans = len(anchors)

    feature_shape = (fsize, osize, osize, ans, 5 + nclasses)
    conv_output = tf.reshape(features, feature_shape)

    all_outputs = tf.split(conv_output, (2, 2, 1, nclasses), axis=-1)
    raw_dxdy, raw_dwdh, raw_conf, raw_prob = all_outputs

    xy_grid = tf.meshgrid(tf.range(osize), tf.range(osize))
    xy_grid = tf.expand_dims(tf.stack(xy_grid, axis=-1), axis=2)
    xy_grid = tf.expand_dims(xy_grid, axis=0)
    xy_grid = tf.cast(tf.tile(xy_grid, [fsize, 1, 1, ans, 1]), tf.float32)

    xy_correction = (tf.sigmoid(raw_dxdy) * xyscale) - 0.5 * (xyscale - 1)
    pred_xy = (xy_correction + xy_grid) * strides
    pred_wh = tf.exp(raw_dwdh) * tf.constant(anchors, dtype=tf.float32)
    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

    pred_conf = tf.sigmoid(raw_conf)
    pred_prob = tf.sigmoid(raw_prob)

    return conv_output, tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)

class Yolo():
    def __init__(self, network, nclasses):
        self._nclasses = nclasses
        self._input_size = get_image_shape(network)[1]
        self._layers = network['layers']

        self._branches = []

        for branch in yolo_outputs(network):
            d_info = [branch[k] for k in ['strides', 'anchors', 'xyscale']]
            self._branches.append((branch['input'], d_info))

    def decode_outputs(self, features, d_info):
        return yolo_decode(features, d_info, self._input_size, self._nclasses)

    def __call__(self, inputs):
        predictions = []
        outputs = build_graph(self._layers, LAYER_FUNCTIONS, inputs)

        for i, decoding_info in self._branches:
            ith_output = outputs[i].output
            predicted_values = self.decode_outputs(ith_output, decoding_info)
            predictions.append(predicted_values)

        return predictions

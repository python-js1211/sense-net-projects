import sensenet.importers
tf = sensenet.importers.import_tensorflow()

from sensenet.constants import MEAN, STANDARD_DEVIATION, ZERO, ONE
from sensenet.layers.utils import constant

class NumericPreprocessor():
    def __init__(self, preprocessor):
        self._moments = [preprocessor[MEAN], preprocessor[STANDARD_DEVIATION]]

        # This should only happen if the feature had a constant value
        # in training
        if self._moments[1] == 0:
            self._moments[1] = 1

    def __call__(self, inputs):
        mean = constant(self._moments[0])
        stdev = constant(self._moments[1])
        output = (inputs - mean) / stdev
        return tf.cast(tf.reshape(output, [-1, 1]), tf.float32)

class BinaryPreprocessor(tf.keras.layers.Layer):
    def __init__(self, preprocessor):
        self._values = [preprocessor[ZERO], preprocessor[ONE]]

    def __call__(self, inputs):
        zero_value = constant(self._values[0])
        output = tf.not_equal(inputs, zero_value)

        return tf.cast(tf.reshape(output, [-1, 1]), tf.float32)

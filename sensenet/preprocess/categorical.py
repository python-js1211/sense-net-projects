import sensenet.importers
tf = sensenet.importers.import_tensorflow()

from sensenet.layers.utils import constant

class CategoricalPreprocessor():
    def __init__(self, preprocessor):
        self._values = preprocessor['values']

    def __call__(self, inputs):
        indices = tf.range(len(self._values), dtype=tf.int32)
        index_col = tf.cast(tf.reshape(inputs, [-1, 1]), tf.int32)
        output = tf.math.equal(index_col, indices)

        return tf.cast(output, tf.float32)

import sensenet.importers
tf = sensenet.importers.import_tensorflow()

from sensenet.layers.utils import constant

class CategoricalPreprocessor():
    def __init__(self, preprocessor):
        self._values = constant([preprocessor['values']], tf.string)

    def __call__(self, inputs):
        output = tf.math.equal(tf.reshape(inputs, [-1, 1]), self._values)
        return tf.cast(output, tf.float32)

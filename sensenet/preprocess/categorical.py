import sensenet.importers
tf = sensenet.importers.import_tensorflow()

from sensenet.layers.utils import constant

MAX_BUCKETS = 1024 * 1024

class CategoricalPreprocessor():
    def __init__(self, preprocessor):
        cats = preprocessor['values']
        buckets = 2
        hashes = tf.strings.to_hash_bucket(cats, buckets).numpy().tolist()

        while len(hashes) != len(set(hashes)) and len(hashes) < MAX_BUCKETS:
            buckets = buckets * 3
            hashes = tf.strings.to_hash_bucket(cats, buckets).numpy().tolist()

        self._buckets = buckets

        if len(hashes) != len(set(hashes)):
            self._hashes = tf.constant(sorted(set(hashes)), dtype=tf.int32)
        else:
            self._hashes = tf.constant(hashes, dtype=tf.int32)

    def __call__(self, inputs):
        hashed = tf.cast(tf.strings.to_hash_bucket(inputs, self._buckets), tf.int32)
        output = tf.math.equal(tf.reshape(hashed, [-1, 1]), self._hashes)
        return tf.cast(output, tf.float32)

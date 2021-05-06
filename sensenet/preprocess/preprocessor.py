import sensenet.importers
tf = sensenet.importers.import_tensorflow()

from sensenet.constants import NUMERIC, CATEGORICAL, IMAGE_PATH
from sensenet.constants import MEAN, STANDARD_DEVIATION
from sensenet.constants import STRING_INPUTS, NUMERIC_INPUTS

from sensenet.preprocess.categorical import CategoricalPreprocessor
from sensenet.preprocess.image import ImagePreprocessor

class Preprocessor():
    def __init__(self, model, extras):
        if model.get('image_network', None) is not None:
            img_proc = ImagePreprocessor(model['image_network'], extras)
            self._image_preprocessor = img_proc

        self._feature_blocks = []

        means = []
        stdevs = []
        block_start = None

        # pprint(model['preprocess'])

        for i, pp in enumerate(model['preprocess']):
            ptype = pp['type']

            if ptype == NUMERIC:
                if block_start is None:
                    block_start = i

                if MEAN in pp:
                    mean, stdev = pp[MEAN], pp[STANDARD_DEVIATION]
                elif 'zero_value' in pp:
                    mean = pp['zero_value']
                    stdev = pp['one_value'] - pp['zero_value']

                # This should only happen if the feature had a
                # constant value in training (and is therefore always
                # zero when standardized)
                if stdev == 0.0:
                    stdev = 1.0

                means.append(mean)
                stdevs.append(stdev)

            elif ptype in [IMAGE_PATH, CATEGORICAL]:
                if block_start is not None:
                    self._feature_blocks.append([block_start, i])

                means.append(0.0)
                stdevs.append(1.0)
                block_start = None

                if ptype == IMAGE_PATH:
                    self._feature_blocks.append([i, self._image_preprocessor])
                else:
                    self._feature_blocks.append([i, CategoricalPreprocessor(pp)])
            else:
                raise ValueError('Cannot make processor with type "%s"' % ptype)

        if block_start is not None:
            self._feature_blocks.append([block_start, len(model['preprocess'])])

        self._means = tf.constant(means)
        self._stdevs = tf.constant(stdevs)

    def __call__(self, inputs):
        str_idx = 0
        processed = []

        try:
            string_inputs = inputs[STRING_INPUTS]
            numeric_inputs = inputs[NUMERIC_INPUTS]
        except TypeError:
            string_inputs = None
            numeric_inputs = inputs

        if len(numeric_inputs.shape) == 2:
            standardized_inputs = (numeric_inputs - self._means) / self._stdevs
        else:
            standardized_inputs = None

        for i, block in enumerate(self._feature_blocks):
            if isinstance(block[1], int):
                start, end = block
                processed.append(standardized_inputs[:, start:end])
            else:
                idx, processor = block

                if isinstance(processor, ImagePreprocessor):
                    if string_inputs is not None:
                        ith_string = tf.reshape(string_inputs[:,str_idx], (-1,))
                        processed.append(processor(ith_string))
                        str_idx += 1
                    else:
                        processed.append(processor(numeric_inputs))
                elif isinstance(processor, CategoricalPreprocessor):
                    cat_idxs = tf.reshape(numeric_inputs[:, idx], (-1,))
                    processed.append(processor(cat_idxs))

        if len(processed) > 1:
            return tf.cast(tf.concat(processed, -1), tf.float32)
        else:
            return tf.cast(processed[0], tf.float32)

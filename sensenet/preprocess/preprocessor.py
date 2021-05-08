import sensenet.importers
tf = sensenet.importers.import_tensorflow()

from sensenet.constants import NUMERIC, CATEGORICAL, IMAGE
from sensenet.constants import MEAN, STANDARD_DEVIATION
from sensenet.constants import PIXEL_INPUTS, NUMERIC_INPUTS

from sensenet.load import count_types
from sensenet.preprocess.categorical import CategoricalPreprocessor
from sensenet.preprocess.image import ImagePreprocessor

class Preprocessor():
    def __init__(self, model, extras):
        if model.get('image_network', None) is not None:
            img_proc = ImagePreprocessor(model['image_network'], extras)
            self._image_preprocessor = img_proc
        else:
            self._image_preprocessor = None

        self._ncolumns, self._nimages = count_types(model['preprocess'])
        self._feature_blocks = []

        means = []
        stdevs = []
        block_start = None

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

            elif ptype in [IMAGE, CATEGORICAL]:
                if block_start is not None:
                    self._feature_blocks.append([block_start, i])

                means.append(0.0)
                stdevs.append(1.0)
                block_start = None

                if ptype == IMAGE:
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
        img_idx = 0
        processed = []

        if isinstance(inputs, dict):
            pixel_inputs = inputs[PIXEL_INPUTS]
            numeric_inputs = inputs[NUMERIC_INPUTS]
        elif self._image_preprocessor is not None:
            pixel_inputs = inputs
            numeric_inputs = None
        else:
            pixel_inputs = None
            numeric_inputs = inputs

        if numeric_inputs is not None:
            standardized_inputs = (numeric_inputs - self._means) / self._stdevs
        else:
            standardized_inputs = None

        for i, block in enumerate(self._feature_blocks):
            if isinstance(block[1], int):
                # This block is a bunch of numeric features that can
                # be passed through without modification
                start, end = block
                processed.append(standardized_inputs[:,start:end])
            else:
                # It's not a numeric feature; there's a processor
                # associated with this column, so process it.
                idx, processor = block

                if isinstance(processor, ImagePreprocessor):
                    if self._nimages == 1:
                        ith_img = pixel_inputs
                    else:
                        ith_img = pixel_inputs[:,img_idx,:,:,:]

                    processed.append(processor(ith_img))
                    img_idx += 1
                elif isinstance(processor, CategoricalPreprocessor):
                    cat_idxs = tf.reshape(numeric_inputs[:,idx], (-1,))
                    processed.append(processor(cat_idxs))

        if len(processed) > 1:
            return tf.cast(tf.concat(processed, -1), tf.float32)
        else:
            return tf.cast(processed[0], tf.float32)

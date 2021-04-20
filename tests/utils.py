import sensenet.importers
tf = sensenet.importers.import_tensorflow()

import os
import gzip
import json

from sensenet.constants import DCT
from sensenet.preprocess.image import rescale

TEST_DATA_DIR = 'tests/data/'
TEST_IMAGE_DATA = os.path.join(TEST_DATA_DIR, 'images/')

def read_regression(path, root=TEST_DATA_DIR):
    return read_zipped_json(os.path.join(root, path))

def read_zipped_json(path):
    with gzip.open(path, "rb") as fin:
        return json.loads(fin.read().decode('utf-8'))

def make_image_reader(settings, target_shape):
    n_chan = target_shape[-1]
    input_format = settings.input_image_format or 'file'
    prefix = settings.image_path_prefix or '.'

    def read_image(path_or_bytes):
        if input_format == 'pixel_values':
            raw = path_or_bytes
        else:
            if input_format == 'image_bytes':
                img_bytes = path_or_bytes
            else:
                path = tf.strings.join([prefix + os.sep, path_or_bytes])
                img_bytes = tf.io.read_file(path)

            raw = tf.io.decode_jpeg(img_bytes, dct_method=DCT, channels=n_chan)

        return rescale(settings, target_shape, raw)

    return read_image

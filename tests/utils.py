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

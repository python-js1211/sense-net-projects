import sensenet.importers
tf = sensenet.importers.import_tensorflow()

import os
import json
import hashlib

from sensenet.constants import CROP, WARP, PAD
from sensenet.models.wrappers import create_model
from sensenet.layers.extract import extract_one, name_index, index_in_model
from sensenet.layers.construct import remove_weights

FILE_PREFIX = 'res'
DEFAULT_START = ['ZeroPadding2D', 0]
DEFAULT_END = ['GlobalAveragePooling2D', -1]

IMAGE_LAYER_STARTS = {
    'mobilenetv2': ['Conv2D', 0],
    'xception': ['Conv2D', 0]
}

IMAGE_LAYER_ENDS = {
    'tinyyolov4': ['Conv2D', -1],
    'yolov4': ['Conv2D', -1]
}

TEMP_WEIGHTS = 'weights.h5'
FEATURIZER_FORMAT = '%s_extractor_%s.h5'
WEIGHTS_FORMAT = '%s_weights_%s.h5'

ALL_NETWORKS = [
    'yolov4',
    'mobilenet',
    'mobilenetv2',
    'resnet18',
    'resnet50',
    'xception',
    'tinyyolov4'
]

def finalize(layers):
    for layer in layers[1:]:
        layer['inputs'] = [name_index(layers, n) for n in layer['input_names']]

    for layer in layers:
        layer.pop('input_names')
        layer.pop('name')

    return layers

def extract(model, start_type_and_idx, end_type_and_idx):
    configs = model.get_config()['layers']
    layers = []

    if start_type_and_idx:
        start_type, start_nth = start_type_and_idx
        start = index_in_model(model, start_type, start_nth)
    else:
        start = 1

    if end_type_and_idx:
        end_type, end_nth = end_type_and_idx
        end = index_in_model(model, end_type, end_nth) + 1
    else:
        end = len(configs)

    for config in configs[start:end]:
        new_layer = extract_one(model, config)
        layers.append(new_layer)

    return finalize(layers)

def extract_image_layers(keras_model, network_name):
    start = IMAGE_LAYER_STARTS.get(network_name, DEFAULT_START)
    end = IMAGE_LAYER_ENDS.get(network_name, DEFAULT_END)

    return extract(keras_model, start, end)

def file_hash(filename):
    sha256_hash = hashlib.sha256()

    with open(filename, 'rb') as fin:
        for byte_block in iter(lambda: fin.read(4096), b''):
            sha256_hash.update(byte_block)

    return sha256_hash.hexdigest()[:8]

def generate_artifacts(network_name):
    print('Reading...')
    with open(os.path.join(FILE_PREFIX, network_name + '.json'), 'r') as fin:
        shell = json.load(fin)

    settings = {
        'input_image_format': 'pixel_values',
        'rescale_type': CROP
    }

    print('Creating model...')
    model = create_model(shell, settings)._model
    image_layers = extract_image_layers(model, network_name)

    print('Saving weights...')
    model.save_weights(TEMP_WEIGHTS)
    version = file_hash(TEMP_WEIGHTS)
    os.rename(TEMP_WEIGHTS, WEIGHTS_FORMAT % (network_name, version))

    print('Writing featurizer...')
    if 'yolo' not in network_name:
        shell['layers'] = remove_weights(extract(model, ['Dense', 0], None))
        end_idx = index_in_model(model, 'GlobalAveragePooling2D', -1)
        f_end = model.layers[end_idx].output

        featurizer = tf.keras.Model(inputs=model.layers[0].input, outputs=f_end)
        featurizer.save(FEATURIZER_FORMAT % (network_name, version))
    else:
        shell['layers'] = []

    shell['image_network']['layers'] = remove_weights(image_layers)
    shell['image_network']['metadata']['version'] = version
    shell['image_network']['metadata'].pop('mean_image', None)
    shell['image_network']['metadata']['rescale_type'] = CROP

    print('Done.')
    return shell

def all_artifacts():
    outmap = {}

    for network in ALL_NETWORKS:
        print(network)
        shell = generate_artifacts(network)
        outmap[network] = shell

    return outmap

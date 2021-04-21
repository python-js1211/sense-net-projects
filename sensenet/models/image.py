import sensenet.importers
tf = sensenet.importers.import_tensorflow()

import math

from sensenet.accessors import is_yolo_model
from sensenet.layers.extract import name_index, input_indices, make_layer_map
from sensenet.models.bounding_box import box_detector
from sensenet.models.deepnet import deepnet_model
from sensenet.models.settings import ensure_settings
from sensenet.preprocess.preprocessor import Preprocessor
from sensenet.pretrained import load_pretrained_weights, get_pretrained_network

START_TYPES = ['ZeroPadding2D', 'Conv2D']
END_TYPES = ['GlobalAveragePooling2D', 'GlobalMaxPooling2D', 'Conv2D']

YOLO_N_CONV = 110
TINYYOLO_N_CONV = 21

def image_model(network, input_settings):
    settings = ensure_settings(input_settings)

    if is_yolo_model(network):
        model = box_detector(network, settings)
    else:
        model = deepnet_model(network, settings)

    if settings.load_pretrained_weights:
        load_pretrained_weights(model, network['image_network'])

    return model

def pretrained_image_model(network_name, settings):
    network = get_pretrained_network(network_name)
    return image_model(network, settings)

def io_for_extractor(model):
    image_layers = get_image_layers(model, truncate_start=False)
    return image_layers[0].input, image_layers[-1].output

def image_feature_extractor(model):
    image_input, features = io_for_extractor(model)
    return tf.keras.Model(inputs=image_input, outputs=features)

def get_image_layer_boundary(model, first):
    layers = model.get_config()['layers']
    matching = []

    if first:
        types = START_TYPES
        nth = 0
    else:
        types = END_TYPES
        nth = -1

    for layer in layers:
        if layer['class_name'] in types:
            matching.append(layer)

    if not matching:
        raise ValueError('%s not found in model' % ltype)
    else:
        if first:
            return matching[nth]['name']
        elif len(matching) == YOLO_N_CONV:
            return [m['name'] for m in matching[-3:]]
        elif len(matching) == TINYYOLO_N_CONV:
            return [m['name'] for m in matching[-2:]]
        else:
            return [matching[nth]['name']]

def get_image_layers(model, truncate_start=True):
    all_layers = model.get_config()['layers']
    layer_map = make_layer_map(model)
    end_names = get_image_layer_boundary(model, False)
    all_indices = set()

    for end_name in end_names:
        input_indices(layer_map, end_name, all_indices)

    indices = sorted(all_indices)

    if truncate_start:
        begin_name = get_image_layer_boundary(model, True)
        begin_index = name_index(all_layers, begin_name)
        image_layer_indices = indices[indices.index(begin_index):]
    else:
        image_layer_indices = indices

    return [model.get_layer(index=i) for i in image_layer_indices]

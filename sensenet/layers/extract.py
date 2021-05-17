IGNORED_LAYERS = ['dropout']

def add(config, layer):
    return {'type': 'add'}

def activation(config, layer):
    return {
        'type': 'activation',
        'activation_function': layer.activation.__name__
    }

def batchnorm(config, layer):
    gamma, beta, mean, variance = layer.get_weights()

    return {
        'type': 'batch_normalization',
        'epsilon': config['epsilon'],
        'gamma': gamma.tolist(),
        'beta': beta.tolist(),
        'mean': mean.tolist(),
        'variance': variance.tolist()
    }

def concat(config, layer):
    return {'type': 'concatenate'}

def conv_2d(config, layer):
    if config['use_bias']:
        kernel, bias = layer.get_weights()
    else:
        kernel = layer.get_weights()[0]
        bias = None

    return {
        'type': 'convolution_2d',
        'kernel': kernel.tolist(),
        'bias': bias.tolist() if bias is not None else None,
        'activation_function': config['activation'],
        'padding': config['padding'],
        'strides': config['strides']
    }

def dense(config, layer):
    if config['use_bias']:
        weights, offset = layer.get_weights()
    else:
        weights = layer.get_weights()[0]
        offset = None

    return {
        'type': 'dense',
        'weights': weights.tolist(),
        'offset': offset.tolist() if offset is not None else None,
        'activation_function': config['activation']
    }

def depthwise_conv_2d(config, layer):
    if config['use_bias']:
        kernel, bias = layer.get_weights()
    else:
        kernel = layer.get_weights()[0]
        bias = None

    return {
        'type': 'depthwise_convolution_2d',
        'kernel': kernel.tolist(),
        'bias': bias.tolist() if bias is not None else None,
        'activation_function': config['activation'],
        'padding': config['padding'],
        'strides': config['strides'],
        'depth_multiplier': config['depth_multiplier']
    }

def global_max_pool(config, layer):
    return {'type': 'global_max_pool_2d'}

def global_avg_pool(config, layer):
    return {'type': 'global_average_pool_2d'}

def dropout(config, layer):
    return {'type': 'dropout'}

def lamda(config, layer):
    fname = layer.function.__name__

    if fname.startswith('split_'):
        pieces = fname.split('_')
        ith = int(pieces[1])
        nsplits = int(pieces[3])

        return {
            'type': 'split_channels',
            'number_of_splits': nsplits,
            'group_index': ith
        }
    else:
        raise ValueError('Cannot serialize lambda with function %s' % fname)

def max_pool(config, layer):
    return {
        'type': 'max_pool_2d',
        'padding': config['padding'],
        'strides': config['strides'],
        'pool_size': config['pool_size']
    }

def separable_conv_2d(config, layer):
    if config['use_bias']:
        depth_kernel, point_kernel, bias = layer.get_weights()
    else:
        depth_kernel, point_kernel = layer.get_weights()[:2]
        bias = None

    return {
        'type': 'separable_convolution_2d',
        'depth_kernel': depth_kernel.tolist(),
        'point_kernel': point_kernel.tolist(),
        'bias': bias.tolist() if bias is not None else None,
        'activation_function': config['activation'],
        'padding': config['padding'],
        'strides': config['strides'],
        'depth_multiplier': config['depth_multiplier']
    }

def upsample(config, layer):
    return {
        'type': 'upsampling_2d',
        'method': 'bilinear',
        'size': [2, 2]
    }

def zero_pad(config, layer):
    return {'type': 'padding_2d', 'padding': config['padding']}

LAYER_EXTRACTORS = {
    'Activation': activation,
    'Add': add,
    'BatchNormalization': batchnorm,
    'Concatenate': concat,
    'Conv2D': conv_2d,
    'Dense': dense,
    'DepthwiseConv2D': depthwise_conv_2d,
    'Dropout': dropout,
    'AlphaDropout': dropout,
    'GlobalAveragePooling2D': global_avg_pool,
    'GlobalMaxPooling2D': global_max_pool,
    'Lambda': lamda,
    'MaxPooling2D': max_pool,
    'SeparableConv2D': separable_conv_2d,
    'UpSampling2D': upsample,
    'ZeroPadding2D': zero_pad
}

def index_in_model(model, ltype, nth):
    layers = model.get_config()['layers']
    matching = []

    for i, layer in enumerate(layers):
        if layer['class_name'] == ltype:
            matching.append(i)

    if not matching:
        raise ValueError('%s not found in model' % ltype)
    else:
        return matching[nth]

def input_indices(layer_map, layer_name, index_set):
    index_set.add(layer_map[layer_name]['index'])

    if layer_map[layer_name]['inbound_nodes']:
        if isinstance(layer_map[layer_name]['inbound_nodes'][0][0], list):
            for in_layer in layer_map[layer_name]['inbound_nodes'][0]:
                in_name = in_layer[0]

                if layer_map[in_name]['index'] not in index_set:
                    input_indices(layer_map, in_name, index_set)
        else:
            in_name = layer_map[layer_name]['inbound_nodes'][0][0]

            if layer_map[in_name]['index'] not in index_set:
                input_indices(layer_map, in_name, index_set)

    return index_set

def name_index(layers, name):
    for i, layer in enumerate(layers):
        if layer['name'] == name:
            return i

    raise ValueError('%s not found in layer stack' % name)

def make_layer_map(model):
    all_layers = model.get_config()['layers']
    layer_map = {}

    for i, layer in enumerate(all_layers):
        layer_map[layer['name']] = layer
        layer_map[layer['name']]['index'] = i

    return layer_map

def extract_one(layer_map, layer):
    config = layer_map[layer.name]

    try:
        processor = LAYER_EXTRACTORS[config['class_name']]
    except KeyError:
        raise ValueError('No processor for type %s' % config['class_name'])

    new_layer = processor(config['config'], layer)
    new_layer['name'] = config['name']
    new_layer['input_names'] = [n[0] for n in config['inbound_nodes'][0]]

    return new_layer

def filter_ignored(layers):
    for layer in layers[1:]:
        for i, in_name in enumerate(layer['input_names']):
            input_layer = layers[name_index(layers, in_name)]

            if 'type' in input_layer and input_layer['type'] in IGNORED_LAYERS:
                # Dropout layer; the previous layer is the actual input
                assert len(input_layer['input_names']) == 1
                layer['input_names'][i] = input_layer['input_names'][0]

    return list(filter(lambda l: l['type'] not in IGNORED_LAYERS, layers))

def extract_layers_list(model, keras_layers):
    all_layers = []
    layer_map = make_layer_map(model)

    for layer in keras_layers:
        new_layer = extract_one(layer_map, layer)
        all_layers.append(new_layer)

    output_layers = filter_ignored(all_layers)

    for layer in output_layers[1:]:
        input_names = layer['input_names']
        layer['inputs'] = [name_index(output_layers, n) for n in input_names]

    for layer in output_layers:
        layer.pop('input_names')
        layer.pop('name')

    return output_layers

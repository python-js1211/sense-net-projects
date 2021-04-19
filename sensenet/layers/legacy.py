import sensenet.importers
np = sensenet.importers.import_numpy()
tf = sensenet.importers.import_tensorflow()
kl = sensenet.importers.import_keras_layers()

from sensenet.layers.utils import transpose, activation_function, get_units

def dense_with_weights(params):
    imap = {}

    for key in ['weights', 'offset']:
        if isinstance(params[key], str):
            imap[key] = params[key]
        else:
            imap[key] = tf.constant_initializer(np.array(params[key]))

    return kl.Dense(get_units(params),
                    dtype=tf.float32,
                    activation=activation_function(params),
                    use_bias=True,
                    kernel_initializer=imap['weights'],
                    bias_initializer=imap['offset'])

def batchnorm_with_weights(params):
    imap = {}

    for key in ['beta', 'gamma', 'mean', 'variance']:
        if isinstance(params[key], str):
            imap[key] = params[key]
        else:
            imap[key] = tf.constant_initializer(np.array(params[key]))

    return kl.BatchNormalization(dtype=tf.float32,
                                 epsilon=params.get('epsilon', 1e-3),
                                 beta_initializer=imap['beta'],
                                 gamma_initializer=imap['gamma'],
                                 moving_mean_initializer=imap['mean'],
                                 moving_variance_initializer=imap['variance'])

def with_popped_activation(params):
    afn = activation_function(params)
    params_copy = dict(params)
    params_copy.pop('activation_function', None)

    return params_copy, afn

class LegacyBlock():
    def __init__(self, params):
        super(LegacyBlock, self).__init__()

        dense_params, afn = with_popped_activation(params)

        self._dense = dense_with_weights(dense_params)
        self._bnorm = batchnorm_with_weights(params)
        self._activator = kl.Activation(afn)

    def __call__(self, inputs):
        propigated = self._dense(inputs)
        normalized = self._bnorm(propigated)

        return self._activator(normalized)

def legacy(params):
    dense_params = dict(params)

    if not isinstance(params['weights'], str):
        # Needs converting from old format
        dense_params['weights'] = transpose(dense_params['weights'])

    dense_params['beta'] = dense_params['offset']
    dense_params['gamma'] = dense_params['scale']

    if params['stdev'] is not None:
        if isinstance(params['stdev'], str):
            dense_params['variance'] = 'ones'
            dense_params['offset'] = 'zeros'
        else:
            variance = np.square(np.array(params['stdev'])) - 1e-3
            dense_params['variance'] = variance.tolist()
            dense_params['offset'] = np.zeros(np.array(variance.shape)).tolist()

        return LegacyBlock(dense_params)
    else:
        return dense_with_weights(dense_params)

class LegacyResidualBlock():
    def __init__(self, params_list):
        super(LegacyResidualBlock, self).__init__()

        first_params = params_list[0]
        second_params, afn = with_popped_activation(params_list[1])

        self._first = legacy(first_params)
        self._second = legacy(second_params)
        self._activator = kl.Activation(afn)

    def equalize_input_width(self, inputs, outputs):
        if inputs.shape[1] == outputs.shape[1]:
            return inputs
        elif inputs.shape[1] > outputs.shape[1]:
            return inputs[:,:outputs.shape[1]]
        else:  # need to repeat inputs until we get to the output shape
            to_concat = []
            ncols = 0

            while ncols < outputs.shape[1]:
                if ncols + inputs.shape[1] < outputs.shape[1]:
                    to_concat.append(inputs)
                    ncols += inputs.shape[1]
                else:
                    to_add = outputs.shape[1] - ncols
                    to_concat.append(inputs[:,:to_add])
                    ncols += to_add

            return tf.concat(to_concat, -1)

    def __call__(self, inputs):
        first_out = self._first(inputs)
        residuals = self._second(first_out)

        tiled_inputs = self.equalize_input_width(inputs, residuals)
        outputs = tiled_inputs + residuals

        return self._activator(outputs)

def build_legacy_graph(layers_params, initial_inputs):
    layers = []
    use_next = True

    for i, lp in enumerate(layers_params):
        if use_next:
            if i + 1 < len(layers_params):
                residuals = layers_params[i + 1].get('residuals', False)
            else:
                residuals = False

            if residuals:
                layer = LegacyResidualBlock([lp, layers_params[i + 1]])
                use_next = False
            else:
                layer = legacy(lp)

            if len(layers) == 0:
                next_inputs = layer(initial_inputs)
            else:
                next_inputs = layer(next_inputs)

            layers.append(layer)
        else:
            use_next = True

    return layers

def to_legacy_residual(block):
    dpath = list(block['dense_path'])

    if dpath[-1]['type'] == 'batch_normalization':
        dpath.append({'type': 'activation', 'activation_function': 'linear'})

    out_layers = to_legacy_sequence(dpath)

    out_layers[-1]['activation_function'] = block['activation_function']
    out_layers[-1]['residuals'] = True

    return out_layers, 1

def to_legacy_batchnorm(layer):
    variance = np.array(layer['variance'])

    out_layer = {
        'weights': np.eye(variance.shape[0]).tolist(),
        'offset': layer['beta'],
        'scale': layer['gamma'],
        'mean': layer['mean'],
        'stdev': np.sqrt(variance + 1e-3).tolist(),
        'residuals': False,
        'activation_function': 'identity'
    }

    return [out_layer], 1

def to_legacy_layer(layers, i):
    assert layers[i]['type'] == 'dense'

    out_layer = {
        'weights': transpose(layers[i]['weights']),
        'offset': layers[i]['offset'],
        'scale': None,
        'mean': None,
        'stdev': None,
        'residuals': False,
        'activation_function': layers[i]['activation_function']
    }

    nlayers = 1
    afn = out_layer['activation_function']

    if i + 1 < len(layers) and layers[i + 1]['type'] == 'activation':
        assert afn in [None, 'identity', 'linear'], afn

        nlayers = 2
        out_layer['activation_function'] = layers[i + 1]['activation_function']
    if i + 2 < len(layers):
        if (layers[i + 1]['type'] == 'batch_normalization' and
            layers[i + 2]['type'] == 'activation'):

            assert afn in [None, 'identity', 'linear'], afn

            nlayers = 3
            offset = np.array(out_layer['offset'])
            batch_layer = to_legacy_batchnorm(layers[i + 1])[0][0]
            batch_mean = np.array(batch_layer['mean'])

            out_layer.update(batch_layer)
            out_layer['weights'] = transpose(layers[i]['weights'])
            out_layer['mean'] = (batch_mean - offset).tolist()
            out_layer['activation_function'] = layers[i + 2]['activation_function']

    return [out_layer], nlayers

def to_legacy_sequence(layers):
    i = 0
    out_layers = []

    while i < len(layers):
        ltype = layers[i].get('type', 'fully_connected')

        if ltype == 'dense_residual_block':
            outputs, nlayers = to_legacy_residual(layers[i])
        elif ltype == 'dense':
            outputs, nlayers = to_legacy_layer(layers, i)
        elif ltype == 'batch_normalization':
            outputs, nlayers = to_legacy_batchnorm(layers[i])
        elif ltype == 'fully_connected':
            outputs, nlayers = ([layers[i]], 1)
        else:
            raise ValueError('Type is %s' % ltype)

        out_layers.extend(outputs)
        i += nlayers

    return out_layers

def legacy_convert(model_json):
    output = dict(model_json)

    if 'layers' in model_json:
        output['layers'] = to_legacy_sequence(model_json['layers'])
    elif 'networks' in model_json:
        outnets = []

        for network in model_json['networks']:
            newnet = dict(network)
            newnet['layers'] = to_legacy_sequence(network['layers'])
            outnets.append(newnet)

        output['networks'] = outnets
    else:
        raise ValueError('Wrong format: %s' % sorted(model_json.keys()))

    return output

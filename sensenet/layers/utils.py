import sensenet.importers
np = sensenet.importers.import_numpy()
tf = sensenet.importers.import_tensorflow()
kl = sensenet.importers.import_keras_layers()

from sensenet.constants import LEAKY_RELU_ALPHA

# Custom activations need to be mapped to named functions, so when we
# extract them from the model we can grab the function's name with
# `Layer.function.__name__`
def leaky_relu(x):
    return tf.nn.leaky_relu(x, alpha=LEAKY_RELU_ALPHA)

def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))

def relu6(x):
    return tf.nn.relu6(x)

ACTIVATORS = {
    'leaky_relu': leaky_relu,
    'mish': mish,
    'relu6': relu6
}

# The names of the trainable parameters in any layer
WEIGHT_INITIALIZERS = {
    'weights': 'glorot_uniform',
    'offset': 'zeros',
    'gamma': 'ones',
    'scale': 'ones',
    'beta': 'zeros',
    'mean': 'zeros',
    'variance': 'ones',
    'stdev': 'ones',
    'kernel': 'glorot_uniform',
    'bias': 'zeros',
    'depth_kernel': 'glorot_uniform',
    'point_kernel': 'glorot_uniform'
}

WEIGHT_KEYS = {
    'BatchNormalization': ['gamma', 'beta', 'mean', 'variance'],
    'Conv2D': ['kernel', 'bias'],
    'Dense': ['weights', 'offset'],
    'DepthwiseConv2D': ['kernel', 'bias'],
    'SeparableConv2D': ['depth_kernel', 'point_kernel', 'bias']
}

INITIALIZERS = {
    'glorot_uniform': tf.initializers.glorot_uniform,
    'glorot_normal': tf.initializers.glorot_normal
}

def log_summary(x, msg):
    def summary_function(x):
        summary = [tf.shape(x), tf.reduce_mean(x), tf.math.reduce_std(x)]
        return tf.compat.v1.Print(x, summary, summarize=1024, message=msg)

    return kl.Lambda(summary_function)(x)

def variable(value, is_training, datatype=tf.float32):
    return tf.Variable(initial_value=value,
                       trainable=is_training,
                       dtype=datatype)

def constant(value, datatype=tf.float32):
    return tf.constant(value, dtype=datatype)

def transpose(amatrix):
    arr = np.array(amatrix)
    return np.transpose(arr).tolist()

def shape(tensor):
    return np.array(tensor.get_shape().as_list(), dtype=np.float32)

def get_units(params):
    if isinstance(params['weights'], str):
        return int(params['number_of_nodes'])
    else:
        return len(params['weights'][0])

def activation_function(params):
    afn = params.get('activation_function', None)

    if afn in [None, 'linear', 'identity']:
        return 'linear'
    elif afn in ACTIVATORS:
        return ACTIVATORS[afn]
    else:
        return params['activation_function']

def initializer_map(params):
    imap = {}

    for k in WEIGHT_INITIALIZERS:
        weights = params.get(k, None)

        if isinstance(weights, str):
            if weights in ['zeros', 'ones']:
                imap[k] = weights
            else:
                random_seed = params.get('seed', 0)
                assert random_seed is not None
                imap[k] = INITIALIZERS[weights](seed=random_seed)
        else:
            imap[k] = 'zeros'

    return imap

def build_graph(layers_params, creation_functions, initial_inputs):
    outputs = []
    layers = []

    if layers_params and len(layers_params) > 0:
        for params in layers_params:
            layer = creation_functions[params['type']](params)
            input_idxs = params.get('inputs', [-1])

            if len(outputs) == 0:
                next_inputs = layer(initial_inputs)
            elif len(input_idxs) == 1:
                next_inputs = layer(outputs[input_idxs[0]])
            else:
                next_inputs = layer([outputs[idx] for idx in input_idxs])

            type_name = type(layer).__name__

            if type_name in WEIGHT_KEYS:
                weights = []

                for i, key in enumerate(WEIGHT_KEYS[type_name]):
                    if params.get(key, None) is None:
                        if type_name == 'Dense' and key == 'offset':
                            pval = np.zeros(layer.get_weights()[i].shape)
                        else:
                            pval = None
                    elif not isinstance(params[key], str):
                        pval = np.array(params[key])
                    else:
                        pval = None

                    if pval is not None:
                        weights.append(pval)

                if weights:
                    layer.set_weights(weights)

            outputs.append(next_inputs)
            layers.append(layer)

    return layers

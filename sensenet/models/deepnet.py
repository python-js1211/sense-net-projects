import sensenet.importers
tf = sensenet.importers.import_tensorflow()
kl = sensenet.importers.import_keras_layers()

from sensenet.constants import IMAGE_PATH, NUMERIC, CATEGORICAL
from sensenet.accessors import get_output_exposition
from sensenet.layers.utils import propagate
from sensenet.layers.tree import ForestPreprocessor
from sensenet.layers.construct import layer_sequence, tree_preprocessor
from sensenet.models.settings import ensure_settings
from sensenet.preprocess.preprocessor import Preprocessor
from sensenet.pretrained import load_pretrained_weights

def instantiate_inputs(model, settings):
    preprocessors = model['preprocess']
    ptypes = [p['type'] for p in preprocessors]

    ncols = len(preprocessors)
    nstrings = ptypes.count(IMAGE_PATH) + ptypes.count(CATEGORICAL)

    if ncols == 1 and settings.input_image_format == 'pixel_values':
        assert ptypes[0] == IMAGE_PATH

        shp = model['image_network']['metadata']['input_image_shape']
        return kl.Input((shp[1], shp[0], shp[2]), dtype=tf.float32, name='image')
    else:
        return {
            'numeric': kl.Input((ncols,), dtype=tf.float32, name='numeric'),
            'string': kl.Input((nstrings,), dtype=tf.string, name='string')
        }

def apply_layers(model, inputs, treeed_inputs):
    if 'networks' in model:
        all_layer_sequences = [layer_sequence(net) for net in model['networks']]
        use_trees = [net.get('trees', False) for net in model['networks']]
    elif model['layers']:
        all_layer_sequences = [layer_sequence(model)]
        use_trees = [treeed_inputs is not None]
    else:
        all_layer_sequences = [[]]
        use_trees = [False]

    all_predictions = []
    outex = get_output_exposition(model)

    for lseq, tree_in in zip(all_layer_sequences, use_trees):
        if tree_in:
            preds = propagate(lseq, treeed_inputs)
        else:
            preds = propagate(lseq, inputs)

        if outex['type'] == NUMERIC:
            preds = preds * outex['stdev'] + outex['mean']

        all_predictions.append(preds)

    if len(all_predictions) > 1:
        summed = tf.add_n(all_predictions)
        predictions = summed / len(all_predictions)
    else:
        predictions = all_predictions[0]

    return predictions

def deepnet_model(model, input_settings):
    settings = ensure_settings(input_settings)

    preprocessor = Preprocessor(model, settings)
    trees = tree_preprocessor(model)

    raw_inputs = instantiate_inputs(model, settings)
    inputs = preprocessor(raw_inputs)

    if trees:
        treeed_inputs = trees(inputs)
    else:
        treeed_inputs = None

    predictions = apply_layers(model, inputs, treeed_inputs)
    return tf.keras.Model(inputs=raw_inputs, outputs=predictions)

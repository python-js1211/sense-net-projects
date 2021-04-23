import sensenet.importers
tf = sensenet.importers.import_tensorflow()
kl = sensenet.importers.import_keras_layers()

from sensenet.accessors import get_image_shape, get_output_exposition
from sensenet.accessors import get_image_tensor_shape
from sensenet.constants import IMAGE_PATH, NUMERIC, CATEGORICAL
from sensenet.constants import STRING_INPUTS, NUMERIC_INPUTS, PIXEL_INPUTS
from sensenet.layers.construct import feed_through, tree_preprocessor
from sensenet.load import load_points
from sensenet.models.settings import ensure_settings
from sensenet.preprocess.preprocessor import Preprocessor

def instantiate_inputs(model, settings):
    preprocessors = model['preprocess']
    ptypes = [p['type'] for p in preprocessors]

    ncols = len(preprocessors)
    nstrings = ptypes.count(IMAGE_PATH)

    if ncols == 1 and settings.input_image_format == 'pixel_values':
        assert ptypes[0] == IMAGE_PATH
        image_shape = get_image_tensor_shape(settings)

        return kl.Input(image_shape,
                        dtype=tf.float32,
                        name=PIXEL_INPUTS)
    else:
        return {
            NUMERIC_INPUTS: kl.Input((ncols,),
                                     dtype=tf.float32,
                                     name=NUMERIC_INPUTS),
            STRING_INPUTS: kl.Input((nstrings,),
                                    dtype=tf.string,
                                    name=STRING_INPUTS)
        }

def apply_layers(model, settings, inputs, treeed_inputs):
    if 'networks' in model:
        all_layer_sequences = [net['layers'] for net in model['networks']]
        use_trees = [net.get('trees', False) for net in model['networks']]
    elif model['layers']:
        all_layer_sequences = [model['layers']]
        use_trees = [treeed_inputs is not None]
    else:
        all_layer_sequences = [[]]
        use_trees = [False]

    all_predictions = []
    outex = get_output_exposition(model)

    for lseq, tree_in in zip(all_layer_sequences, use_trees):
        if tree_in:
            preds = feed_through(lseq, treeed_inputs)
        else:
            preds = feed_through(lseq, inputs)

        if outex['type'] == NUMERIC and not settings.regression_normalize:
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

    predictions = apply_layers(model, settings, inputs, treeed_inputs)
    return tf.keras.Model(inputs=raw_inputs, outputs=predictions)

import os

import sensenet.importers
np = sensenet.importers.import_numpy()
tf = sensenet.importers.import_tensorflow()

from sensenet.constants import NUMERIC, CATEGORICAL
from sensenet.preprocess.preprocessor import Preprocessor
from sensenet.layers.legacy import legacy_convert
from sensenet.layers.tree import ForestPreprocessor
from sensenet.load import load_points
from sensenet.models.deepnet import DeepnetWrapper
from sensenet.models.settings import Settings
from sensenet.io.save import assets_for_deepnet, write_model

from .utils import TEST_DATA_DIR, read_regression

EMBEDDING = 'embedding.json.gz'
SIMPLE = 'regression.json.gz'
LEGACY = 'legacy_regression.json.gz'
SEARCH = 'search_regression.json.gz'
LEGACY_SEARCH = 'legacy_search_regression.json.gz'
IMAGE = 'image_regression.json.gz'

EXTRA_PARAMS = Settings({'image_path_prefix': TEST_DATA_DIR + 'images/digits/'})

def validate_predictions(test_artifact):
    test_model, test_points = [test_artifact[k] for k in ['model', 'validation']]

    # Make sure we have a reasonable set of test points
    assert len(test_points) > 4
    assert not any([t == test_points[0] for t in test_points[1:]])

    ins = [t['input'] for t in test_points]
    outs = np.array([t['output'] for t in test_points])

    model = DeepnetWrapper(test_model, EXTRA_PARAMS)

    converted = legacy_convert(test_model)
    legacy_model = DeepnetWrapper(converted, EXTRA_PARAMS)

    for i, true_pred in enumerate(outs):
        mod_pred = model.predict([ins[i]])
        legacy_pred = legacy_model.predict(ins[i])

        outstr = '\nPred: %s\nExpt: %s' % (str(mod_pred[0]), str(true_pred))
        legstr = '\nLegacy: %s\nExpt: %s' % (str(legacy_pred[0]), str(true_pred))

        assert np.allclose(mod_pred[0], true_pred, atol=1e-7), outstr
        assert np.allclose(legacy_pred[0], true_pred, atol=1e-7), legstr

    mod_preds = model.predict(ins)
    assert np.allclose(mod_preds, outs, atol=1e-7)

def single_artifact(regression_path, index):
    test_artifact = read_regression(regression_path)[index]
    validate_predictions(test_artifact)

def test_simple_networks():
    rdata = read_regression(SIMPLE)

    for i in range(len(rdata)):
        yield single_artifact, SIMPLE, i

def test_legacy_networks():
    rdata = read_regression(LEGACY)

    for i in range(len(rdata)):
        yield single_artifact, LEGACY, i

def test_one():
    single_artifact(SIMPLE, 0)

def fake_outex(test_info):
    anode = test_info['trees'][0][1][0]
    while len(anode) > 2:
        anode = anode[2]

    nouts = len(anode[0])

    if nouts > 1:
        return {'type': CATEGORICAL, 'values': ['v%d' for v in range(nouts)]}
    else:
        return {'type': NUMERIC, 'mean': 0, 'stdev': 1}

def single_embedding(index):
    test_info = read_regression(EMBEDDING)[index]
    test_info['output_exposition'] = fake_outex(test_info)

    preprocessor = Preprocessor(test_info, EXTRA_PARAMS)
    forest = ForestPreprocessor(test_info)

    inputs = load_points(test_info['preprocess'], test_info['input_data'])
    proc_result = preprocessor(inputs)
    tree_result = forest(proc_result)

    data_with_trees = np.array(test_info['with_trees'])
    data_without_trees = np.array(test_info['without_trees'])

    assert np.allclose(proc_result, data_without_trees)
    assert np.allclose(tree_result, data_with_trees)

def test_embedding():
    artifact = read_regression(EMBEDDING)

    for i in range(len(artifact)):
        yield single_embedding, i

def test_search():
    test_artifact = read_regression(SEARCH)[0]
    validate_predictions(test_artifact)

def test_legacy_search():
    test_artifact = read_regression(LEGACY_SEARCH)
    validate_predictions(test_artifact)

def test_images():
    test_artifact = read_regression(IMAGE)[0]
    validate_predictions(test_artifact)

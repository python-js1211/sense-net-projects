import sensenet.importers
np = sensenet.importers.import_numpy()
tf = sensenet.importers.import_tensorflow()

import os

from PIL import Image

from sensenet.constants import CATEGORICAL, IMAGE, BOUNDING_BOX
from sensenet.constants import NUMERIC_INPUTS, WARP, CROP

from sensenet.accessors import get_image_shape
from sensenet.layers.extract import extract_layers_list
from sensenet.load import load_points
from sensenet.models.bundle import write_bundle
from sensenet.models.image import pretrained_image_model, image_feature_extractor
from sensenet.models.image import get_image_layers
from sensenet.models.settings import Settings, ensure_settings
from sensenet.models.wrappers import create_model
from sensenet.preprocess.image import make_image_reader
from sensenet.pretrained import get_extractor_bundle
from sensenet.pretrained import get_pretrained_network, get_extractor_bundle

from .utils import TEST_DATA_DIR, TEST_IMAGE_DATA

EXTRA_PARAMS = {
    'bounding_box_threshold': 0.5,
    'load_pretrained_weights': True,
    'rescale_type': CROP
}

CLASSIFIER_TEST_IMAGES = [('dog.jpg', 254), ('bus.jpg', 779)]
EXTRACTOR_FORMAT = '%s_extractor_%s'

def write_bundle_file(tf_model, metadata):
    network = metadata['base_image_network']
    version = metadata['version']
    name = EXTRACTOR_FORMAT % (network, version)

    model_path = os.path.join(TEST_DATA_DIR, name)
    os.makedirs(model_path)
    tf_model.save(model_path)

    write_bundle(model_path)

def create_image_model(network_name, additional_settings):
    extras = dict(EXTRA_PARAMS)

    if additional_settings:
        extras.update(additional_settings)

    return pretrained_image_model(network_name, Settings(extras))

def reader_for_network(network_name, additional_settings):
    extras = dict(EXTRA_PARAMS)

    if additional_settings:
        extras.update(additional_settings)

    settings = Settings(extras)
    image_shape = get_image_shape(get_pretrained_network(network_name))
    return make_image_reader('file', image_shape, TEST_IMAGE_DATA, settings)

def check_image_prediction(prediction, index, pos_threshold, neg_threshold):
    for i, p in enumerate(prediction.flatten().tolist()):
        if i == index:
            assert p > pos_threshold, str((i, p))
        else:
            assert p < neg_threshold, str((i, p))

def classify(network_name, accuracy_threshold):
    network = get_pretrained_network(network_name)
    nlayers = len(network['image_network']['layers'])
    noutputs = network['image_network']['metadata']['outputs']
    preprocessors = network['preprocess']

    pixel_model = create_image_model(network_name, None)

    assert len(get_image_layers(pixel_model)) == nlayers

    for image, cidx in CLASSIFIER_TEST_IMAGES:
        image_path = os.path.join(TEST_IMAGE_DATA, image)
        point = load_points(preprocessors, [[image_path]])
        pred = pixel_model.predict(point)

        check_image_prediction(pred, cidx, accuracy_threshold, 0.02)

    ex_mod = image_feature_extractor(pixel_model)
    bundle_mod = create_model(get_extractor_bundle(network_name))

    read = reader_for_network(network_name, None)
    img_arrays = np.array([read(im[0]).numpy() for im in CLASSIFIER_TEST_IMAGES])

    bundle_outputs = bundle_mod(img_arrays)
    ex_outputs = ex_mod(img_arrays)

    assert ex_outputs.shape == (2, noutputs)
    assert bundle_outputs.shape == (2, noutputs)

    abs_out = np.abs(ex_outputs - bundle_outputs)
    assert np.mean(abs_out) < 1e-5, abs_out

def test_resnet50():
    classify('resnet50', 0.99)

def test_mobilenet():
    classify('mobilenet', 0.97)

def test_xception():
    classify('xception', 0.88)

def test_resnet18():
    classify('resnet18', 0.96)

def test_mobilenetv2():
    classify('mobilenetv2', 0.87)

def detect_bounding_boxes(network_name, nboxes, class_list, threshold):
    extras = {'bounding_box_threshold': threshold}

    network = get_pretrained_network(network_name)
    nlayers = len(network['image_network']['layers'])

    detector = create_image_model(network_name, extras)
    read = reader_for_network(network_name, {'rescale_type': 'pad'})

    image_layers = get_image_layers(detector)
    ex_layers = extract_layers_list(detector, image_layers)

    assert len(image_layers) == len(ex_layers) == nlayers

    img_px = np.expand_dims(read('pizza_people.jpg').numpy(), axis=0)
    pred = detector.predict(img_px)

    boxes, scores, classes = pred[0][0], pred[1][0], pred[2][0]

    assert len(boxes) == len(scores) == nboxes, len(boxes)
    assert sorted(set(classes)) == sorted(class_list), classes

def test_tinyyolov4():
    detect_bounding_boxes('tinyyolov4', 5, [0, 53], 0.4)

def test_yolov4():
    detect_bounding_boxes('yolov4', 8, [0, 41, 53, 60], 0.5)

def test_empty():
    detector = create_image_model('tinyyolov4', None)
    image_path = os.path.join(TEST_IMAGE_DATA, 'black.png')
    image = load_points([{'type': IMAGE, 'index': 0}], [[image_path]])
    boxes, scores, classes  = detector.predict(image)

    assert len(boxes[0]) == 0
    assert len(scores[0]) == 0
    assert len(classes[0]) == 0

def test_scaling():
    detector = create_image_model('tinyyolov4', None)
    image_path = os.path.join(TEST_IMAGE_DATA, 'strange_car.png')
    image = load_points([{'type': IMAGE, 'index': 0}], [[image_path]])
    boxes, scores, classes  = detector.predict(image)

    assert 550 < boxes[0, 0, 0] < 600,  boxes[0, 0]
    assert 220 < boxes[0, 0, 1] < 270,  boxes[0, 0]
    assert 970 < boxes[0, 0, 2] < 1020,  boxes[0, 0]
    assert 390 < boxes[0, 0, 3] < 440,  boxes[0, 0]

    assert scores[0] > 0.9
    assert classes[0] == 2

def test_black_and_white():
    pixel_model = create_image_model('mobilenetv2', None)
    image_path = os.path.join(TEST_IMAGE_DATA, 'model_t.jpg')

    point = load_points([{'type': IMAGE, 'index': 0}], [[image_path]])
    pred = pixel_model.predict(point)
    check_image_prediction(pred, 661, 0.95, 0.02)

    with Image.open(image_path) as img:
        image_array = np.array(img)


    point = load_points([{'type': IMAGE, 'index': 0}], [[image_array]])
    pred = pixel_model.predict(point)
    check_image_prediction(pred, 661, 0.95, 0.02)

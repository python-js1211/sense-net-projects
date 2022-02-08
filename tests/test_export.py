import sensenet.importers

np = sensenet.importers.import_numpy()
tf = sensenet.importers.import_tensorflow()

import os
import shutil
import gzip
import json

from PIL import Image

from sensenet.constants import WARP
from sensenet.models.wrappers import Deepnet, ObjectDetector
from sensenet.models.wrappers import convert, tflite_predict

from .utils import TEST_DATA_DIR, TEST_IMAGE_DATA
from .test_pretrained import create_image_model

MOBILENET_PATH = os.path.join(TEST_DATA_DIR, "mobilenetv2.json.gz")
TEST_SAVE_MODEL = os.path.join(TEST_DATA_DIR, "test_model_save")
TEST_TF_LITE_MODEL = os.path.join(TEST_SAVE_MODEL, "model.tflite")


def make_classifier(network_name):
    model = create_image_model(network_name, None)
    return Deepnet(model, None)


def make_detector(network_name, unfiltered):
    filtering = {"output_unfiltered_boxes": unfiltered}

    if unfiltered:
        filtering["rescale_type"] = WARP

    model = create_image_model(network_name, filtering)

    return ObjectDetector(model, filtering)


def convert_and_predict(model, test_file):
    shutil.rmtree(TEST_SAVE_MODEL, ignore_errors=True)
    os.makedirs(TEST_SAVE_MODEL)

    convert(model, None, TEST_TF_LITE_MODEL, "tflite")
    img_path = os.path.join(TEST_IMAGE_DATA, test_file)

    return tflite_predict(TEST_TF_LITE_MODEL, img_path)


def test_tflite_deepnet():
    probs = convert_and_predict(make_classifier("mobilenetv2"), "dog.jpg")

    assert len(probs) == 1
    assert probs[0].shape == (1, 1000), probs[0].shape
    # It's generous but the math here is 32-bit
    assert abs(np.sum(probs[0]) - 1) < 1e-5, np.sum(probs[0])
    assert probs[0][0, 254] > 0.87, probs[0][0, 254]

    shutil.rmtree(TEST_SAVE_MODEL)


def test_tflite_boxes():
    detector = make_detector("tinyyolov4", True)
    scores, boxes, classes = convert_and_predict(detector, "square_car.png")

    assert boxes.shape == (1, 2535, 4), boxes.shape
    assert classes.shape == (1, 2535), classes.shape
    assert scores.shape == (1, 2535), scores.shape

    nboxes = 0

    for box, cls, score in zip(boxes[0], classes[0], scores[0]):
        # There are a few boxes here, but they should all find the
        # same thing in roughly the same place
        if score > 0.5:
            assert 20 < box[0] < 70, (box, cls, score)
            assert 250 < box[1] < 300, (box, cls, score)
            assert 440 < box[2] < 490, (box, cls, score)
            assert 390 < box[3] < 440, (box, cls, score)

            assert cls == 2

            nboxes += 1

    assert nboxes > 1

    shutil.rmtree(TEST_SAVE_MODEL)


def test_tfjs_classifier():
    shutil.rmtree(TEST_SAVE_MODEL, ignore_errors=True)
    convert(make_classifier("mobilenetv2"), None, TEST_SAVE_MODEL, "tfjs")

    # Commenting this delete out will allow you to test the exported
    # model with nodejs/tensorflowjs/canvas, if you have these things
    # installed, using the test_model.js script in this directory, by
    # navigating to sensenet/tests and running `node test_model.js`.
    shutil.rmtree(TEST_SAVE_MODEL, ignore_errors=True)


def test_tfjs_boxes():
    shutil.rmtree(TEST_SAVE_MODEL, ignore_errors=True)

    detector = make_detector("tinyyolov4", False)
    convert(detector, None, TEST_SAVE_MODEL, "tfjs")

    # As above, you can comment this out to test in JS, but here you
    # must also set the `bounding_boxes` variable to true in the test
    # script.
    shutil.rmtree(TEST_SAVE_MODEL, ignore_errors=True)


def test_all_conversions():
    with gzip.open(MOBILENET_PATH, "rb") as fin:
        jmodel = json.load(fin)

    for aformat in ["tflite", "tfjs", "smbundle", "h5"]:
        outpath = TEST_SAVE_MODEL + "." + aformat
        convert(jmodel, None, outpath, aformat)

        if aformat == "tfjs":
            shutil.rmtree(outpath)
        else:
            os.remove(outpath)

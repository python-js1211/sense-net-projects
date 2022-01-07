import sensenet.importers

np = sensenet.importers.import_numpy()
tf = sensenet.importers.import_tensorflow()

import os
import json
import gzip

from PIL import Image

from sensenet.constants import DCT
from sensenet.load import to_image_pixels
from sensenet.models.wrappers import create_model

from .utils import TEST_DATA_DIR, TEST_IMAGE_DATA
from .test_pretrained import check_image_prediction

BUS_INDEX = 779

BUS_PATH = os.path.join(TEST_IMAGE_DATA, "bus.jpg")
TIFF_PATH = os.path.join(TEST_IMAGE_DATA, "dog.tiff")
TINYYOLO_MODEL_PATH = os.path.join(TEST_DATA_DIR, "test_model.json.gz")
MOBILENET_PATH = os.path.join(TEST_DATA_DIR, "mobilenetv2.json.gz")
TEXT_MODEL_PATH = os.path.join(TEST_DATA_DIR, "text_model.json.gz")
TWO_FIELD_MODEL_PATH = os.path.join(TEST_DATA_DIR, "image_and_category.json.gz")


def make_mobilenet(settings):
    with gzip.open(MOBILENET_PATH, "rb") as fin:
        network = json.load(fin)

    return create_model(network, settings)


def check_pixels_and_file(settings, pos_threshold, neg_threshold):
    pixel_model = make_mobilenet(settings)

    with Image.open(BUS_PATH) as img:
        if settings.get("color_space", "").endswith("a"):
            image_pixels = np.array(img.convert("RGBA"))
            new_settings = dict(settings)
            new_settings["color_space"] = settings["color_space"][:3]
            file_model = make_mobilenet(new_settings)
        else:
            image_pixels = np.array(img)
            file_model = pixel_model

    file_pred = file_model(BUS_PATH)
    pixel_pred = pixel_model([[image_pixels]])

    diffs = np.abs(pixel_pred - file_pred)
    assert np.all(diffs < 1e-3), np.max(diffs)

    for pred in [pixel_pred, file_pred]:
        check_image_prediction(pred, BUS_INDEX, pos_threshold, neg_threshold)

    return pixel_pred


def test_channel_order():
    for space, pos, neg in [("rgb", 0.98, 0.02), ("bgr", 0.11, 0.5)]:
        settings = {"color_space": space}
        no_alpha = check_pixels_and_file(settings, pos, neg)

        settings = {"color_space": space + "a"}
        alpha = check_pixels_and_file(settings, pos, neg)

        assert np.all(np.abs(alpha - no_alpha) < 1e-4)


def test_cropping():
    for rt, threshold in [("warp", 0.98), ("pad", 0.96), ("crop", 0.98)]:
        settings = {"rescale_type": rt}
        pred = check_pixels_and_file(settings, threshold, 0.02)
        pred_value = pred[0, BUS_INDEX]

        assert threshold < pred_value < threshold + 0.01, pred_value


def test_ndarray():
    pixel_model = make_mobilenet(None)

    with Image.open(BUS_PATH) as img:
        image_pixels = np.array(img)

    good_float_pixels = image_pixels.astype(np.float64) / 255.0

    bad_float_pixels = np.array(good_float_pixels, copy=True)
    bad_float_pixels[200, 200, 0] = 257

    pred = pixel_model(good_float_pixels)
    check_image_prediction(pred, BUS_INDEX, 0.98, 0.02)

    try:
        pixel_model(bad_float_pixels)
        assert False, 'This "image" should throw an exception'
    except ValueError:
        pass


def test_text():
    with gzip.open(TEXT_MODEL_PATH, "rb") as fin:
        network = json.load(fin)

    text_model = create_model(network)
    assert text_model._model is not None


def test_image_plus_categorical():
    with gzip.open(TWO_FIELD_MODEL_PATH, "rb") as fin:
        network = json.load(fin)

    model = create_model(network)

    # Just test we're getting values that are numeric and reasonable
    assert 0 < model([BUS_PATH, "f1"])[0][0] < 10
    assert 0 < model([BUS_PATH, None])[0][0] < 10
    assert 0 < model([None, None])[0][0] < 10


def test_detect():
    with gzip.open(TINYYOLO_MODEL_PATH, "rb") as fin:
        network = json.load(fin)

    model = create_model(network)
    image_path = os.path.join(TEST_IMAGE_DATA, "tolls.jpg")

    with Image.open(image_path) as img:
        image_pixels = np.array(img)

    assert len(model(image_pixels)) == 0


def test_tiff_image():
    img = to_image_pixels(TIFF_PATH, None)
    assert type(img) == np.ndarray
    assert img.shape == (600, 800, 3)

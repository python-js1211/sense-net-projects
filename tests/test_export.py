import sensenet.importers
np = sensenet.importers.import_numpy()
tf = sensenet.importers.import_tensorflow()

import os
import shutil

from PIL import Image

from sensenet.models.wrappers import convert, Deepnet, ObjectDetector

from .utils import TEST_DATA_DIR, TEST_IMAGE_DATA
from .test_pretrained import create_image_model

TEST_SAVE_MODEL = os.path.join(TEST_DATA_DIR, 'test_model_save')
TEST_TF_LITE_MODEL = os.path.join(TEST_SAVE_MODEL, 'model.tflite')

def make_classifier(network_name):
    model = create_image_model(network_name, None)
    return Deepnet(model, None)

def make_detector(network_name, unfiltered):
    filtering = {'output_unfiltered_boxes': unfiltered}
    model = create_image_model(network_name, filtering)

    return ObjectDetector(model, filtering)

def tflite_predict(model, len_inputs, len_outputs, test_file):
    shutil.rmtree(TEST_SAVE_MODEL, ignore_errors=True)
    os.makedirs(TEST_SAVE_MODEL)

    convert(model, None, TEST_TF_LITE_MODEL, 'tflite')
    img = Image.open(os.path.join(TEST_IMAGE_DATA, test_file))
    in_shape = [1] + list(img.size)[::-1] + [3]

    interpreter = tf.lite.Interpreter(model_path=TEST_TF_LITE_MODEL)
    interpreter.resize_tensor_input(0, in_shape, strict=True)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    assert len(input_details) == len_inputs
    assert len(output_details) == len_outputs

    input_data = np.expand_dims(img.convert('RGB'), axis=0).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    return [interpreter.get_tensor(od['index']) for od in output_details]

def test_tflite_deepnet():
    probs = tflite_predict(make_classifier('mobilenetv2'), 1, 1, 'dog.jpg')

    assert len(probs) == 1
    assert probs[0].shape == (1, 1000), probs[0].shape
    # It's generous but the math here is 32-bit
    assert abs(np.sum(probs[0]) - 1) < 1e-5, np.sum(probs[0])
    assert probs[0][0, 254] > 0.87, probs[0][0, 254]

    shutil.rmtree(TEST_SAVE_MODEL)

def test_tflite_boxes():
    detector = make_detector('tinyyolov4', True)
    boxes, scores, classes = tflite_predict(detector, 1, 3, 'strange_car.png')

    assert boxes.shape == (1, 2535, 4), boxes.shape
    assert classes.shape == (1, 2535), classes.shape
    assert scores.shape == (1, 2535), scores.shape

    for box, cls, score in zip(boxes[0], classes[0], scores[0]):
        # There are a few boxes here, but they should all find the
        # same thing in roughly the same place
        if score > 0.5:
            assert 550 < box[0] < 600, (box, cls, score)
            assert 220 < box[1] < 270, (box, cls, score)
            assert 970 < box[2] < 1020, (box, cls, score)
            assert 390 < box[3] < 440, (box, cls, score)

            assert cls == 2

    shutil.rmtree(TEST_SAVE_MODEL)

def test_tfjs_classifier():
    shutil.rmtree(TEST_SAVE_MODEL, ignore_errors=True)
    convert(make_classifier('mobilenetv2'), None, TEST_SAVE_MODEL, 'tfjs')

    # Commenting this delete out will allow you to test the exported
    # model with nodejs/tensorflowjs/canvas, if you have these things
    # installed, using the test_model.js script in this directory, by
    # navigating to sensenet/tests and running `node test_model.js`.
    shutil.rmtree(TEST_SAVE_MODEL, ignore_errors=True)

def test_tfjs_boxes():
    shutil.rmtree(TEST_SAVE_MODEL, ignore_errors=True)

    detector = make_detector('tinyyolov4', False)
    convert(detector, None, TEST_SAVE_MODEL, 'tfjs')

    # As above, you can comment this out to test in JS, but here you
    # must also set the `bounding_boxes` variable to true in the test
    # script.
    shutil.rmtree(TEST_SAVE_MODEL, ignore_errors=True)

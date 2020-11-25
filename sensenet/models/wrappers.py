import sensenet.importers
np = sensenet.importers.import_numpy()
tf = sensenet.importers.import_tensorflow()

import json

from sensenet.accessors import is_yolo_model, get_output_exposition
from sensenet.load import load_points
from sensenet.models.bounding_box import box_detector
from sensenet.models.deepnet import deepnet_model
from sensenet.models.settings import ensure_settings

def tflite_export(tf_model, model_path):
    converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
    tflite_model = converter.convert()

    if model_path is not None:
        with open(model_path, 'wb') as fout:
            fout.write(tflite_model)

    return tflite_model

def to_tflite(model_or_spec, output_path, sensenet_settings=None):
    """Convert some structure describing a wrapped model to a tflite file.

    The first input to this function can be one of the model wrapper
    classes (e.g., ObjectDetector), a dict used to instantiate a
    wrapper, or the path to a file containing JSON for such a dict.

    The second is the desired path to the tflite output file.

    Optionally, one may provide a dict of settings for the model,
    which can contain things like the IOU threshold for non-max
    suppression (see sensenet.model.settings).  Note that if the input
    is an already-instantiated model, this argument is ignored.

    Returns the converted model in tflite as a `bytes`.

    """
    if isinstance(model_or_spec, tf.keras.Model):
        model = model_or_spec
    elif isinstance(model_or_spec, (Deepnet, ObjectDetector)):
        model = model_or_spec._model
    else:
        if isinstance(model_or_spec, dict):
            model_dict = model_or_spec
        else:
            with open(model_or_spec, 'r') as fin:
                model_dict = json.load(fin)

        export_settings = ensure_settings(sensenet_settings)
        # This is the only valid input format for tflite; it doesn't
        # know how to read files (as of TF 2.3)
        export_settings.input_image_format = 'pixel_values'

        model = create_model(model_dict, settings=export_settings)._model

    return tflite_export(model, output_path)

class Deepnet(object):
    def __init__(self, model, settings):
        self._preprocessors = model['preprocess']
        self._model = deepnet_model(model, settings)

        outex = get_output_exposition(model)

        try:
            self._classes = outex['values']
        except KeyError:
            self._classes = None

    def predict(self, points):
        pvec = load_points(self._preprocessors, points)
        return self._model.predict(pvec)

    def __call__(self, input_data):
        # TODO:  We should probably add assertions or something here
        # to make sure we're getting at least the right number of inputs
        if isinstance(input_data, list):
            if isinstance(input_data[0], (float, int, str)):
                # Single unwrapped instance
                return self.predict([input_data])
            else:
                # Properly wrapped instance
                return self.predict(input_data)
        # Single image path or text field
        elif isinstance(input_data, str):
            return self.predict([[input_data]])
        else:
            dtype = str(type(input_data))
            raise TypeError('Cannot predict on arguments of type "%s"' % dtype)

class ObjectDetector(object):
    def __init__(self, model, settings):
        self._unfiltered = settings.output_unfiltered_boxes
        self._model = box_detector(model, settings)
        self._classes = model['output_exposition']['values']

    def predict(self, points):
        return self._model.predict(points)

    def __call__(self, input_data):
        # Single wrapped instance, or multiple instances
        if isinstance(input_data, list):
            if len(input_data) > 1:
                return [self(data) for data in input_data]
            else:
                prediction = self.predict([input_data])
        # Single image path
        elif isinstance(input_data, str):
            prediction = self.predict([[input_data]])
        # Pixel-valued ndarray input
        elif isinstance(input_data, np.ndarray) and len(input_data.shape) == 3:
            array = np.expand_dims(input_data, axis=0)
            prediction = self.predict(array)
        # Something else (tf.tensor or python list)
        else:
            prediction = self.predict(input_data)

        if self._unfiltered:
            return prediction
        else:
            boxes, scores, classes = prediction
            output_boxes = []

            for box, score, cls in zip(boxes[0], scores[0], classes[0]):
                output_boxes.append({
                    'box': [int(c) for c in box],
                    'label': self._classes[int(cls)],
                    'score': float(score)})

            return output_boxes

def is_deepnet(model):
    try:
        return 'layers' in model or 'networks' in model
    except:
        return False

def bigml_resource(resource):
    if 'deepnet' in resource:
        model = resource['deepnet']
    elif 'model' in resource:
        model = resource['model']
    else:
        model = {}

    try:
        return model['network']
    except:
        return None

def create_model(model, settings=None):
    settings_object = ensure_settings(settings)

    if bigml_resource(model):
        return create_model(bigml_resource(model), settings=settings)
    elif is_deepnet(model):
        if is_yolo_model(model):
            return ObjectDetector(model, settings_object)
        else:
            return Deepnet(model, settings_object)
    elif isinstance(model, dict):
        raise ValueError('Model format not recognized: %s' % str(model.keys()))
    else:
        raise TypeError('`model` argument cannot be a %s' % str(type(model)))

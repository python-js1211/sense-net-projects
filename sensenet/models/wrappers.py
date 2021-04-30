import sensenet.importers
np = sensenet.importers.import_numpy()
tf = sensenet.importers.import_tensorflow()

import json
import os
import shutil
import tempfile
import sys

from contextlib import contextmanager

from sensenet.constants import STRING_INPUTS, NUMERIC_INPUTS
from sensenet.constants import IMAGE_PATH

from sensenet.accessors import is_yolo_model, get_output_exposition
from sensenet.load import load_points
from sensenet.models.bounding_box import box_detector
from sensenet.models.bundle import read_bundle, write_bundle, BUNDLE_EXTENSION
from sensenet.models.deepnet import deepnet_model
from sensenet.models.settings import ensure_settings

SETTINGS_PATH = os.path.join('assets', 'settings.json')

@contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

class SaveableModel(object):
    def __init__(self, keras_model, settings):
        if isinstance(keras_model, tf.keras.Model):
            self._model = keras_model

            for key in settings:
                setattr(self, key, settings[key])

    def save_weights(self, save_path):
        self._model.save_weights(save_path)

    def save_bundle(self, save_path):
        outdir, model_name = os.path.split(save_path)

        assert os.path.exists(outdir) or len(outdir) == 0

        if not model_name:
            raise ValueError('Name "%s" from "%s" is invalid' %
                             (model_name, save_path))

        if save_path.endswith(BUNDLE_EXTENSION):
            out_path = save_path
        else:
            out_path = save_path + BUNDLE_EXTENSION

        assert not os.path.exists(out_path)

        with tempfile.TemporaryDirectory() as saved_model_temp:
            for k, v in self._model._get_trainable_state().items():
                k.trainable = False

            self._model.compile()

            model_path = os.path.join(saved_model_temp, model_name)
            self._model.save(model_path)

            attributes = dict(vars(self))
            attributes.pop('_model')
            attributes['deepnet_type'] = type(self).__name__

            with open(os.path.join(model_path, SETTINGS_PATH), 'w') as fout:
                json.dump(attributes, fout)

            bundle_file = write_bundle(model_path)
            os.rename(bundle_file, out_path)

    def save_tflite(self, save_path):
        converter = tf.lite.TFLiteConverter.from_keras_model(self._model)
        tflite_model = converter.convert()

        with open(save_path, 'wb') as fout:
            fout.write(tflite_model)

        return tflite_model

    def save_tfjs(self, save_path):
        # Leave this import unless we absolutely need it
        import tensorflowjs as tfjs

        with tempfile.TemporaryDirectory() as saved_model_temp:
            self._model.save(saved_model_temp)
            with suppress_stdout():
                tfjs.converters.convert_tf_saved_model(saved_model_temp,
                                                       save_path,
                                                       skip_op_check=True)

class Deepnet(SaveableModel):
    def __init__(self, model, settings):
        super().__init__(model, settings)

        if isinstance(model, dict):
            outex = get_output_exposition(model)

            try:
                self._classes = outex['values']
            except KeyError:
                self._classes = None

            self._preprocessors = model['preprocess']
            self._model = deepnet_model(model, settings)

        # Pretrained image networks should be the only thing missing
        # this `_preprocessors` attribute
        pps = getattr(self, '_preprocessors', None)

        if pps is None or (len(pps) == 1 and pps[0]['type'] == IMAGE_PATH):
            self._single_image = True
        else:
            self._single_image = False

    def load_and_predict(self, points):
        pvec = load_points(self._preprocessors, points)
        return self._model.predict(pvec)

    def __call__(self, input_data):
        # TODO:  We should probably add assertions or something here
        # to make sure we're getting at least the right number of inputs
        if isinstance(input_data, list):
            if isinstance(input_data[0], (float, int, str)):
                # Single unwrapped instance
                return self.load_and_predict([input_data])
            else:
                # Properly wrapped instance
                return self.load_and_predict(input_data)
        elif isinstance(input_data, np.ndarray):
            # Pixel-valued ndarray input image; will only work for
            # single images
            if self._single_image:
                assert len(input_data.shape) in [3, 4]
                if len(input_data.shape) == 3:
                    array = np.expand_dims(input_data, axis=0)
                else:
                    array = input_data
            else:
                assert len(input_data.shape) in [1, 2]
                if len(input_data.shape) == 1:
                    numeric = np.expand_dims(input_data, axis=0)
                else:
                    numeric = input_data

                array = {
                    NUMERIC_INPUTS: numeric,
                    STRING_INPUTS: np.zeros((numeric.shape[0],0))
                }

            return self._model.predict(array)
        # Single image path
        elif isinstance(input_data, str):
            if self._single_image:
                return self.load_and_predict([[input_data]])
            else:
                raise ValueError('Single strings not accepted as input')
        else:
            dtype = str(type(input_data))
            raise TypeError('Cannot predict on arguments of type "%s"' % dtype)

class ObjectDetector(SaveableModel):
    def __init__(self, model, settings):
        super().__init__(model, settings)

        if isinstance(model, dict):
            self._model = box_detector(model, settings)
            self._classes = model['output_exposition']['values']
            self._unfiltered = settings.output_unfiltered_boxes
        elif 'output_unfiltered_boxes' in settings:
            self._unfiltered = settings['output_unfiltered_boxes']

    def __call__(self, input_data):
        # Single wrapped instance, or multiple instances
        if isinstance(input_data, list):
            if len(input_data) > 1:
                return [self(data) for data in input_data]
            else:
                prediction = self._model.predict([input_data])
        # Single image path
        elif isinstance(input_data, str):
            prediction = self._model.predict([[input_data]])
        # Pixel-valued ndarray input
        elif isinstance(input_data, np.ndarray):
            if len(input_data.shape) == 3:
                array = np.expand_dims(input_data, axis=0)
            else:
                array = input_data

            prediction = self._model.predict(array)
        # Something else (tf.tensor or python list)
        else:
            prediction = self._model.predict(input_data)

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

def model_from_dictionary(model_dict, settings):
    settings_object = ensure_settings(settings)

    if bigml_resource(model_dict):
        model = bigml_resource(model)
    else:
        model = model_dict

    if is_deepnet(model):
        if is_yolo_model(model):
            return ObjectDetector(model, settings_object)
        else:
            return Deepnet(model, settings_object)
    elif isinstance(model, dict):
        raise ValueError('Model format not recognized: %s' % str(model.keys()))

def model_from_bundle(bundle_file):
    bundle_name = os.path.basename(bundle_file)

    with tempfile.TemporaryDirectory() as saved_model_temp:
        temp_bundle = os.path.join(saved_model_temp, bundle_name)
        shutil.copyfile(bundle_file, temp_bundle)
        model_dir = read_bundle(temp_bundle)

        model = tf.keras.models.load_model(model_dir)
        settings_path = os.path.join(model_dir, SETTINGS_PATH)

        if os.path.exists(settings_path):
            with open(settings_path, 'r') as fin:
                settings = json.load(fin)
        else:
            settings = None

    dtype = settings['deepnet_type']

    if dtype == 'ObjectDetector':
        return ObjectDetector(model, settings)
    elif dtype == 'Deepnet':
        return Deepnet(model, settings)
    else:
        raise ValueError('Invalid deepnet type: "%s"' % dtype)

def create_model(anobject, settings=None):
    if isinstance(anobject, str):
        if os.path.exists(anobject):
            if anobject.endswith(BUNDLE_EXTENSION):
                return model_from_bundle(anobject)
            else:
                with open(anobject, 'r') as fin:
                    return model_from_dictionary(json.load(fin), settings)
        else:
            raise IOError('File %s not found' % str(anobject))
    elif isinstance(anobject, dict):
        return model_from_dictionary(anobject, settings)
    else:
        raise TypeError('Input argument cannot be a %s' % str(type(anobject)))

def convert(model, settings, output_path, to_format):
    """Convert some structure describing a wrapped model to a given output
    format.

    The first input to this function can be one of the model wrapper
    classes (e.g., ObjectDetector), a dict used to instantiate a
    wrapper, the path to a file containing JSON for such a dict, or
    the path to a bundled saved model.

    The second a dict of settings for the model, which can contain
    things like the IOU threshold for non-max suppression (see
    sensenet.model.settings).  Note that if the first argument is an
    already-instantiated model, this argument is ignored.

    The third is the path to which to output the converted model.

    The fourth is the format to which to convert the model, any of
    `smbundle`, `tflite`, `tfjs`, or `h5`, the latter of which saves
    only the weights of the model in keras h5 format without saving
    the layer configs.

    On completion, the requested file is written to the provided path.

    """
    if isinstance(model, SaveableModel):
        model_object = model
    else:
        model_settings = ensure_settings(settings)

        if to_format in ['tflite', 'tfjs']:
            # This is the only valid input format for tflite; it
            # doesn't know how to read files (as of TF 2.3)
            model_settings.input_image_format = 'pixel_values'

        model_object = create_model(model, settings=model_settings)

    if to_format == 'tflite':
        model_object.save_tflite(output_path)
    elif to_format == 'tfjs':
        model_object.save_tfjs(output_path)
    elif to_format == 'smbundle':
        model_object.save_bundle(output_path)
    elif to_format == 'h5':
        model_object.save_weights(output_path)
    else:
        raise ValueError('Format "%s" unknown' % str(to_format))

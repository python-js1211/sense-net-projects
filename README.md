# BigML Sense/Net

Sense/Net is a BigML interface to Tensorflow, which takes a network
specification as a dictionary (read from BigML's JSON model format)
and instantiates a TensorFlow compute graph based on that
specification.

## Entry Points

The library is meant, in general, to take a BigML model specification
as a JSON document, and an optional map of settings and return a
lightweight wrapper around a `tf.keras.Model` based on these
arguments.  The wrapper creation function can be found in
`sensenet.models.wrappers.create_model`

## Pretrained Networks

Often, BigML trained deepnets will use networks pretrained on
[ImageNet](http://www.image-net.org/) either as a starting point for
fine tuning, or as the base layers under a custom set of readout
layers.  The weights for these networks are stored in a public s3
bucket and downloaded as needed for training or inference (see the
`sensenet.pretrained` module).  If the pretrained weights are never
needed, no downloading occurs.

By default, these are downloaded to and read from the directory
`~/.bigml_sensenet` (which is created if it is not present).  To
change the location of this directory, clients can set the environment
variable `BIGML_SENSENET_CACHE_PATH`.

## Model Instantiation

To instantiate a model, pass the model specification and the dict
of additional, optional settings to `create_model`.  For example:

```
model = create_model(a_dict, settings={'image_path_prefix': 'images/path/'})
```

Again, `a_dict` is typically a downloaded BigML model, read into a
python dictionary via `json.load` or similar.

You may also pass the path to a file containing such a model:

```
model = create_model('model.json', settings=None)
```

For image models, `settings` is either `None` (the default) or a dict
of optional settings which may contain:

### Settings Arguments

- `bounding_box_threshold`: For object detection models only, the
  minimal score that an object can have and still be surfaced to the
  user as part of the output.  The default is 0.5, and lower the score
  will have the effect of more (possibly spurious) boxes identified in
  each input image.

- `color_space`: A string which is one of `['rgb', 'rgba', 'bgr',
  'bgra']`.  The first three letters give the order of the color
  channels (red, blue, and green) in the input tensors that will be
  passed to the model.  The final presence or absence of an `'a'`
  indicates that an alpha channel will be present (which will be
  ignored).  This can be useful to match the color space of the output
  model to that provided by another library, such as open CV.  Note
  that TensorFlow uses RGB ordering by default, and all files read by
  TensorFlow are automatically read as RGB files.  This argument is
  generally only necessary if `input_image_format` is
  `'pixel_values'`, and will possibly break predictions if specified
  when the input is a file.

- `image_path_prefix`: A string directory indicating the path where
  images are to be found for image predictions.  When an image path is
  passed at prediction time, this string is prepended to the given
  path.

- `iou_threshold`: A threshold indicating the amount of overlap boxes
  predicting the same class should have before they are considered to
  be bounding the same object.  The default is 0.5, and lower values
  have the effect of eliminating boxes which would otherwise have been
  surfaced to the user.

- `max_objects`: The maximum number of bounding boxes to return for
  each image.  The default is 32.

- `rescale_type`: A string which is one of `['warp', 'pad', 'crop']`.
  If `'warp'`, input images are scaled to the input dimensions
  specified in the network, and their aspect ratios are *not*
  preserved.  If `'pad'`, the image is resized to the smallest
  dimensions such that the image fits into the input dimensions of the
  network, then padded with constant pixels either below or to the
  right to create an appropriately sized image.  For example, if the
  input dimensions of the network are 100 x 100, and we attempt to
  classify a 300 x 600 image, the image is first rescaled to 50 x 100
  (preserving its aspect ratio) then padded on the right to create a
  100 x 100 image.  If `'crop'`, the image is resized to the smallest
  dimension such that the input dimensions fit in the image, then the
  image is centrally cropped to make the specified sizes.  Using the
  sizes in previous example, the image would be rescaled to 100 x 200
  (preserving its aspect ratio) then cropped by 50 pixels on the top
  and bottom to create a 100 x 100 image.

### Model Formats and Conversion

The canonical format for sensenet models is the JSON format
downloadable from BigML.  However, as the JSON is fairly heavyweight,
time-consuming to parse, and not consumable from certain locations,
SenseNet offers a conversion utility,
`sensenet.models.wrappers.convert`, which takes the JSON format as
input and can output the following formats:

- `tflite` will export the model in the Tensorflow lite format, which
  allows lightweight prediction on mobile devices.

- `tfjs` exports the model to the format read by Tensorflow JS to do
  predictions in the browser and server-side in node.js.

- `smbundle` exports the model to a (proprietary) lightweight wrapper
  around the TensorFlow SavedModel format.  The generated file is a
  concatenation of the files in the SavedModel directory, with some
  additional information written to the `assets` sub-directory.  If
  this file is passed to `create_model`, the bundle is extracted to a
  temporary directory, the model instantiated, and the temporary files
  deleted.  To extract the bundle without instantiating the model, see
  the functions in `sensenet.models.bundle`.

- `h5` exports the model **weights only** to the keras h5 model format
  (i.e., via use of the TensorFlow function
  `tf.keras.Model.save_weights`) To use these, you'd instantiate the
  model from JSON and load the weights separately using the
  corresponding TensorFlow `load_weights` function.

## Usage

Once instantiated, you can use the model to make predictions by using
the returned model as a function, like so:

```
prediction = model([1.0, 2.0, 3.0])
```

The input point or points must be a list (or nested list) containing
the input data for each point, in the order implied by
`model._preprocessors`.  Categorical and image variables should be
passed as strings, where the image is either a path to the image on
disk, or the raw compressed image bytes.

For classification or regression models, the function returns a numpy
array where each row is the model's prediction for each input point.
For classification models, there will be a probability for each class
in each row.  For regression models, each row will contain only a
single entry.

For object detection models, the input should always be a single image
(again, either as a file path, compressed byte string, or an array of
pixel values, depending on the settings map, and the result will be
list of detected boxes, each one represented as a dictionary.  For
example:

```
In [5]: model('pizza_people.jpg')
Out[5]:
[{'box': [16, 317, 283, 414], 'label': 'pizza', 'score': 0.9726969599723816},
 {'box': [323, 274, 414, 332], 'label': 'pizza', 'score': 0.7364346981048584},
 {'box': [158, 29, 400, 327], 'label': 'person', 'score': 0.6204285025596619},
 {'box': [15, 34, 283, 336], 'label': 'person', 'score': 0.5346986055374146},
 {'box': [311, 23, 416, 255], 'label': 'person', 'score': 0.41961848735809326}]
```

The `box` array contains the coordinates of the detected box, as `x1,
y1, x2, y2`, where those coordinates represent the upper-left and
lower-right corners of each bounding box, in a coordinate system with
(0, 0) at the upper-left of the input image.  The `score` is the rough
probability that the object has been correctly identified, and the
`label` is the detected class of the object.

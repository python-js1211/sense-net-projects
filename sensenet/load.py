import sensenet.importers
np = sensenet.importers.import_numpy()
tf = sensenet.importers.import_tensorflow()

import os

from sensenet.constants import NUMERIC, CATEGORICAL, IMAGE, DCT
from sensenet.constants import PIXEL_INPUTS, NUMERIC_INPUTS

def count_types(preprocessors):
    if preprocessors is None:
        # This should only happen for pretrained image networks
        return 1, 1
    else:
        ptypes = [p['type'] for p in preprocessors]
        return len(ptypes), ptypes.count(IMAGE)

def list_index(alist, element):
    try:
        return float(alist.index(element))
    except ValueError:
        return -1

def to_image_pixels(image, shape):
    if image is None:
        img_array = np.zeros((1, 1, 3), dtype=np.uint8)
    elif isinstance(image, np.ndarray):
        if len(image.shape) == 2:
            img_array = np.expand_dims(image, -1)
            img_array = np.tile(img_array, (1, 1, 3))
        elif len(image.shape) != 3:
            raise ValueError('Image array has shape %s' % str(image.shape))
        elif image.shape[-1] == 1: # Grayscale
            img_array = np.tile(image, (1, 1, 3))
        elif image.shape[-1] > 4:
            raise ValueError('Number of channels is %d' % image.shape[-1])
        else:
            img_array = image

        if shape is not None and img_array.shape != shape:
            mismatch = '%s != %s' % (str(img_array.shape), str(shape))
            raise IndexError('Image shapes do not all match: %s', mismatch)

        if img_array.dtype != np.uint8:
            if 0 <= np.min(img_array) <= 1 and 0 <= np.max(img_array) <= 1:
                img_array *= 255.

            bounds = np.min(img_array), np.max(img_array)
            if not all([0 <= bound < 256 for bound in bounds]):
                raise ValueError('Bounds for image array are %s' % str(bounds))

            img_array = img_array.astype(np.uint8)

    elif isinstance(image, str):
        if not os.path.exists(image):
            raise ValueError('File %s not found' % image)
        # Allow tensorflow to read the types it is able to read
        elif any([image.endswith(s) for s in ['.jpg', '.jpeg', '.png']]):
            ibytes = tf.io.read_file(image)
            iten = tf.io.decode_jpeg(ibytes, dct_method=DCT, channels=3)
            img_array = iten.numpy()
        # Use PIL for everything else
        else:
            with pil.Image.open(image) as img:
                img_array = np.array(img.convert('RGB'))
    else:
        raise ValueError('Images cannot be type "%s"' % str(type(image)))

    return img_array

def load_points(preprocessors, points):
    nrows = len(points)
    ncols, nimages = count_types(preprocessors)

    inputs = {
        NUMERIC_INPUTS: np.zeros((nrows, ncols), dtype=np.float32),
        PIXEL_INPUTS: [list() for _ in range(nrows)]
    }

    for i, proc in enumerate(preprocessors):
        pidx = proc['index']
        values = proc.get('values', None)

        if proc['type'] == NUMERIC:
            for j, p in enumerate(points):
                inputs[NUMERIC_INPUTS][j, i] = float(p[pidx])
        elif proc['type'] == CATEGORICAL:
            cats = proc['values']
            for j, p in enumerate(points):
                inputs[NUMERIC_INPUTS][j, i] = list_index(cats, str(p[pidx]))
        elif proc['type'] == IMAGE:
            ishape = None

            for j, p in enumerate(points):
                img_array = to_image_pixels(p[pidx], ishape)
                inputs[PIXEL_INPUTS][j].append(img_array)

                # We require here that all images shapes must match if
                # we're going to do a multi-row prediction; else we
                # wouldn't be able to put all of them into an ndarray
                # in the next step.
                if ishape is None:
                    ishape = img_array.shape
        else:
            raise ValueError('Unknown processor type "%s"' % proc['type'])

    if nimages > 0:
        inputs[PIXEL_INPUTS] = np.array(inputs[PIXEL_INPUTS])
        if nimages == ncols:
            # These models are guaranteed to only have one image per
            # row, so slice that dimension off to get [row, h, w,
            # channels] as usual
            if nimages == 1:
                return inputs[PIXEL_INPUTS][:,0,:,:,:]
            else:
                return inputs[PIXEL_INPUTS]
        else:
            return inputs
    else:
        return inputs[NUMERIC_INPUTS]

from sensenet.constants import NUMERIC, CATEGORICAL, BOUNDING_BOX

def get_image_shape(anobject):
    if type(anobject) == dict:
        if 'image_network' in anobject:
            ishape = anobject['image_network']['metadata']['input_image_shape']
        elif 'metadata' in anobject:
            ishape = anobject['metadata']['input_image_shape']
        else:
            msg = 'Cannot find image shape in dict with keys %s'
            raise ValueError(msg % str(sorted(anobject.keys())))
    elif type(anobject) in [list, tuple]:
        ishape = anobject
    else:
        try:
            ishape = anobject.input_image_shape
        except AttributeError:
            msg = 'Cannot find image shape in objective of type "%s"'
            raise ValueError(msg % type(anobject))

    return [None, ishape[1], ishape[0], ishape[2]]

def get_image_tensor_shape(settings):
    if settings.color_space and settings.color_space.lower()[-1] == 'a':
        nchannels = 4
    else:
        nchannels = 3

    return (None, None, nchannels)

def get_output_exposition(model):
    if 'output_exposition' in model:
        return model['output_exposition']
    elif 'networks' in model:
        first_network = model['networks'][0]
        return get_output_exposition(first_network)
    else:
        raise ValueError('Could not locate output_exposition')

def number_of_classes(model):
    outex = get_output_exposition(model)

    if outex['type'] == NUMERIC:
        return 1
    elif outex['type'] in [CATEGORICAL, BOUNDING_BOX]:
        return len(outex['values'])
    else:
        raise ValueError('Output exposition is type "%s"' % outex['type'])

def is_yolo_model(network):
    image_net = network.get('image_network', None)
    return image_net and 'yolo' in image_net['metadata']['base_image_network']

def yolo_outputs(network):
    if 'image_network' in network:
        outputs = network['image_network']['metadata']['outputs']
    elif 'metadata' in network:
        outputs = network['metadata']['outputs']
    else:
        raise ValueError('No YOLO outputs in dict: %s' % str(network.keys()))

    try:
        assert isinstance(outputs, list) and len(outputs) > 1
    except:
        raise ValueError('"outputs" has type %s' % str(type(outputs)))

    return outputs

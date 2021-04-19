import os
import gzip
import json
import requests

USER_HOME = os.path.expanduser('~')
CACHE_DIRECTORY = os.path.join(USER_HOME, '.bigml_sensenet')
CNN_METADATA_FILE = 'sensenet_metadata.json.gz'

S3_BUCKET = 'https://s3.amazonaws.com/bigml-cnns/'
MIN_NETWORK_SIZE = 128 * 1024

WEIGHTS_FORMAT = '%s_weights_%s.h5'
BUNDLE_FORMAT = '%s_extractor_%s.smbundle'

def read_resource(afile):
    try:
        import importlib.resources as pkg_resources
    except ImportError:
        import importlib_resources as pkg_resources

    cmp_bytes = pkg_resources.read_binary(__package__, afile)
    astr = gzip.decompress(cmp_bytes).decode('utf-8')

    return json.loads(astr)

PRETRAINED_CNN_METADATA = read_resource(CNN_METADATA_FILE)

def download_to_file(network_archive, path):
    print(('Downloading %s...' % (S3_BUCKET + network_archive)))
    req = requests.get(S3_BUCKET + network_archive)

    with open(path, 'wb') as fout:
        fout.write(req.content)

def cache_resource_path(resource_name):
    try:
        os.makedirs(CACHE_DIRECTORY)
    except OSError:
        pass

    cache_path = os.path.join(CACHE_DIRECTORY, resource_name)

    if (not os.path.exists(cache_path)) or os.path.isdir(cache_path):
        download_to_file(resource_name, cache_path)
    elif os.path.getsize(cache_path) < MIN_NETWORK_SIZE:
        download_to_file(resource_name, cache_path)

    if os.path.getsize(cache_path) < MIN_NETWORK_SIZE:
        raise ValueError('File %s looks too small to be correct' % cache_path)

    return cache_path

def get_pretrained_network(network_name):
    if network_name in PRETRAINED_CNN_METADATA:
        return PRETRAINED_CNN_METADATA[network_name]
    else:
        raise KeyError('%s is not a pretrained network' % network_name)

def load_pretrained_weights(model, network):
    meta = network['metadata']
    wpath = WEIGHTS_FORMAT % (meta['base_image_network'], meta['version'])
    archive = cache_resource_path(wpath)

    model.load_weights(archive)

def get_extractor_bundle(network_name):
    meta = get_pretrained_network(network_name)['image_network']['metadata']
    bpath = BUNDLE_FORMAT % (meta['base_image_network'], meta['version'])

    return cache_resource_path(bpath)

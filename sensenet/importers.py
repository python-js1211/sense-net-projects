"""Namespace just for importing other namespaces to avoid spamming of
various messages on import.

"""
import sys
import os
import logging
import warnings
import glob

from sensenet import __tree_ext_prefix__

logging.getLogger("tensorflow").setLevel(logging.ERROR)

warnings.filterwarnings("ignore", message=".*binary incompatibility.*")
warnings.filterwarnings("ignore", message=".*in favour of importlib.*")
warnings.filterwarnings("ignore", message=".*alias for the builtin.*")


import numpy

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=".*as a synonym of type.*")
    warnings.filterwarnings("ignore", message=".*binary incompatibility.*")
    warnings.filterwarnings("ignore", message=".*details.*")

    import tensorflow
    import tensorflow.keras.layers
    import tensorflowjs

bigml_tf_module = None

for path in sys.path:
    treelib = glob.glob(os.path.join(path, ("*%s*" % __tree_ext_prefix__)))
    if treelib:
        bigml_tf_module = tensorflow.load_op_library(treelib[0])


def import_tensorflow():
    return tensorflow


def import_bigml_treelib():
    return bigml_tf_module


def import_keras_layers():
    return tensorflow.keras.layers


def import_numpy():
    return numpy


def import_tfjs():
    return tensorflowjs

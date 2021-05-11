"""Setup for package sensenet
"""

try:
    import tensorflow as tf
except ModuleNotFoundError:
    raise ImportError("Tensorflow is not in the build environment.")

import pkg_resources

from os import path
from setuptools import setup, Extension, find_packages

from sensenet import __version__, __tree_ext_prefix__

here = path.abspath(path.dirname(__file__))

deps = [
    'importlib-resources>=5.1,<5.2',
    'numpy>=1.19,<1.20',
    'pillow>=8.2,<8.3',
    'tensorflow>=2.4,<2.5',
    'tensorflowjs>=3.4,<3.5'
]

# The installation of `tensorflow-gpu` should be specific to canonical
# docker images distributed by the Tensorflow team.  If they've
# installed tensorflow-gpu, we shouldn't try to install tensorflow on
# top of them.
if any(pkg.key == 'tensorflow-gpu' for pkg in pkg_resources.working_set):
    deps = list(filter(lambda d: not d.startswith('tensorflow>='), deps))

# Get the long description from the relevant file
with open(path.join(here, 'README.md'), 'r') as f:
    long_description = f.read()

compile_args = ['-std=c++11', '-fPIC'] + tf.sysconfig.get_compile_flags()

tree_module = Extension(__tree_ext_prefix__,
                        define_macros = [('MAJOR_VERSION', '1'),
                                         ('MINOR_VERSION', '0')],
                        include_dirs=[tf.sysconfig.get_include()],
                        library_dirs=[tf.sysconfig.get_lib()],
                        extra_compile_args=compile_args,
                        extra_link_args=tf.sysconfig.get_link_flags(),
                        sources = ['cpp/tree_op.cc'])
setup(
    name='bigml-sensenet',
    version=__version__,
    author = 'BigML Team',
    author_email = 'team@bigml.com',
    url = 'http://bigml.com/',
    description='Network builder for bigml deepnet topologies',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    package_data={'sensenet': ["sensenet_metadata.json.gz"]},
    ext_modules=[tree_module],
    tests_require=[
        'nose>=1.3,<1.4',
        'scikit-learn>=0.24,<0.25'
    ],
    test_suite='nose.collector',
    install_requires=deps)

import sensenet.importers
np = sensenet.importers.import_numpy()
tf = sensenet.importers.import_tensorflow()
kl = sensenet.importers.import_keras_layers()

from sensenet.layers.block import BlockMaker
from sensenet.layers.utils import build_graph
from sensenet.layers.construct import LAYER_FUNCTIONS
from sensenet.models.settings import Settings
from sensenet.preprocess.image import make_image_reader

def test_create_simple():
    network = {
        'layers': [
            {
                'type': 'dense',
                'number_of_nodes': 8,
                'seed': 42,
                'activation_function': 'leaky_relu',
                'weights': 'glorot_uniform',
                'offset': 'zeros'
            },
            {
                'type': 'dense',
                'number_of_nodes': 16,
                'seed': 43,
                'activation_function': 'softmax',
                'weights': 'glorot_normal',
                'offset': 'ones'
            }
        ]
    }

    inputs = kl.Input((4,), dtype=tf.float32)
    graph = build_graph(network['layers'], LAYER_FUNCTIONS, inputs)

    assert len(graph) == 2

    for sizes, layer in zip([(4, 8), (8, 16)], graph):
        fan_in, fan_out = sizes
        weights = layer.get_weights()

        assert weights[0].shape == sizes
        assert weights[1].shape == (fan_out,), weights[1].shape

        assert len(set(weights[0].flatten())) == fan_in * fan_out
        assert np.all(weights[0] > -2.0)
        assert np.all(weights[0] < 2.0)

        if fan_out == 16:
            assert np.all(weights[1] == 1.0), weights[1]
        else:
            assert np.all(weights[1] == 0.0), weights[1]

def test_create_residual():
    network = {
        'layers': [
            {
                'type': 'dense',
                'number_of_nodes': 6,
                'seed': 42,
                'activation_function': 'leaky_relu',
                'weights': 'glorot_uniform',
                'offset': 'zeros'
            },
            {
                'type': 'dense_residual_block',
                'activation_function': 'softplus',
                'identity_path': [],
                'dense_path': [
                    {
                        'type': 'dense',
                        'number_of_nodes': 6,
                        'seed': 42,
                        'activation_function': None,
                        'weights': 'glorot_uniform',
                        'offset': 'zeros'
                    },
                    {
                        'type': 'batch_normalization',
                        'beta': 'ones',
                        'gamma': 'zeros',
                        'mean': 'ones',
                        'variance': 'zeros'
                    },
                    {
                        'type': 'activation',
                        'activation_function': 'relu',
                    },
                    {
                        'type': 'dense',
                        'number_of_nodes': 6,
                        'seed': 42,
                        'activation_function': None,
                        'weights': 'glorot_uniform',
                        'offset': 'zeros'
                    },
                    {
                        'type': 'batch_normalization',
                        'beta': 'ones',
                        'gamma': 'ones',
                        'mean': 'zeros',
                        'variance': 'zeros'
                    },
                    {
                        'type': 'activation',
                        'activation_function': 'softmax',
                    }
                ]
            },
            {
                'type': 'dense',
                'number_of_nodes': 4,
                'seed': 42,
                'activation_function': 'softmax',
                'weights': 'glorot_uniform',
                'offset': 'zeros'
            }
        ]
    }

    inputs = kl.Input((8,), dtype=tf.float32)
    graph = build_graph(network['layers'], LAYER_FUNCTIONS, inputs)

    assert len(graph) == 3
    assert [kl.Dense, BlockMaker, kl.Dense] == [type(layer) for layer in graph]
    assert graph[-1].output.shape[1] == 4, graph[-1].output.shape

def show_outputs(images, reader, model):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10,10))

    for n, image in enumerate(images[:16]):
        ax = plt.subplot(4, 4, n + 1)
        img = np.expand_dims(reader(image)[0], axis=0)
        img_pred = np.minimum(255, model(img, training=True)[0].numpy())

        plt.imshow(np.minimum(255, img_pred.astype(np.uint8)))
        plt.axis('off')

    plt.show()

def test_dropblock():
    image_shape = (None, 64, 64, 3)
    network = {
        'layers': [{'type': 'dropout',
                    'dropout_type': 'block',
                    'block_size': 7,
                    'rate': 0.1}]
    }

    inputs = kl.Input(image_shape[1:], dtype=tf.float32)
    graph = build_graph(network['layers'], LAYER_FUNCTIONS, inputs)
    model = tf.keras.Model(inputs=inputs, outputs=graph[-1].output)

    reader = make_image_reader('file', image_shape, 'tests/data/images', None)
    pizzas = ['pizza_people.jpg'] * 16

    # show_outputs(pizzas, reader, model)

    for image in pizzas:
        img = np.expand_dims(reader(image).numpy(), axis=0)
        model(img, training=True)

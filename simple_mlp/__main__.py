"""Main module for the application."""

from . import train as t
import random
import math

random.seed(1)

# activation_functions = [lambda x:x, lambda x: max(0, x)]
# activation_functions_derivatives = [
#     lambda x: 1,
#     lambda x: 0 if x < 0 else
# ( 1 if x > 0 else raise_(Exception("nooooooooo"))) ]


def sigmoid(x):
    """Activation function used for this example."""
    return 1 / (1 + math.exp(-x))


def sigmoid_derivative(sigmoud_output):
    """Activation function derivative used for this example."""
    return sigmoud_output * (1 - sigmoud_output)


if __name__ == "__main__":
    train = t.Train()

    input_count = 2
    network_layout = [2, 1]
    layer_count = len(network_layout)
    activation_functions = [sigmoid] * layer_count
    activation_functions_derivatives = [sigmoid_derivative] * layer_count

    train.generate_xor_dataset(input_count)
    print(train.get_dataset())

    train.setup_network(input_count,
                        network_layout,
                        activation_functions,
                        activation_functions_derivatives)
    train.display_network()
    train.chu_chu(0.5, 0.001)
    train.check_chu_chu()

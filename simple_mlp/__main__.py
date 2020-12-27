from . import train as t
from .test import utils



train = t.Train()

input_count = 2
network_layout = [2, 1]
activation_functions = [lambda x:x, lambda x:x, lambda x: max(0, x)]
activation_functions_derivatives = [
    lambda x: 1,
    lambda x: 1,
    lambda x: 0 if x < 0 else ( 1 if x > 0 else raise_(Exception("nooooooooo"))) ]

train.generate_xor_dataset(input_count)
train.setup_network(input_count, network_layout, activation_functions, activation_functions_derivatives)
train.display_network()

train.chu_chu(0.0001, acceptable_squared_mean_error=0.1)

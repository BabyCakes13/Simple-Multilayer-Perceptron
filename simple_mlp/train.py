import itertools
import functools

class Train:
    def __init__(self):
        self.dataset = self.generate_xor_dataset(2)

    def generate_xor_dataset(self, input_size):
        xor_input_combinations = list(itertools.product([0, 1], repeat=input_size))
        xor_dataset = {}

        for combination in xor_input_combinations:
            xor_dataset[combination] = functools.reduce(lambda x, y: x ^ y, combination)

        return xor_dataset

    def setup_network(self, weights, activation_functions):
        pass

    def setup_training(self):
        activation_function = lambda x: x


train = Train()
print(train.dataset)

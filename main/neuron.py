class Neuron:
    def __init__(self, weights, activation_function):
        self.__weights = weights
        self.__activation_function = activation_function

    def compute(self, input):
        sum = 0

        for i, w in zip(input, self.__weights):
            sum += i * w

        output = self.__activation_function(sum)
        return output

    def get_weights(self):
        return self.__weights

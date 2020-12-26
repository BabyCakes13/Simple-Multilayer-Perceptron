class Neuron:
    def __init__(self, weights, activation_function, activation_function_derivative=None):
        try:
            self.__bias = weights[0]
            self.__weights = weights[1:]
        except IndexError as e:
            raise Exception("Weights does not have enough elements to split it into bias and weights.")

        self.__activation_function = activation_function
        self.__activation_function_derivative = activation_function_derivative

        self.__output = None
        self.__delta = None

    def forward_propagate(self, input):
        # insert the input for the bias
        if len(input) != len(self.__weights):
            raise Exception("The number of input is different than the number of weights in {}".format(self))

        sum = 0
        for i, w in zip(input, self.__weights):
            sum += i * w

        sum += self.__bias

        output = self.__activation_function(sum)
        self.__output = output

        return output

    def get_output_derivative(self):
        result = self.__activation_function_derivative(self.__output)
        return result

    def get_weights(self):
        return self.__weights

    def get_output(self):
        return self.__output

    def get_delta(self):
        return self.__delta

    def get_weight(self, index):
        return self.__weights[index]

    def set_output(self, output):
        self.__output = output

    def set_delta(self, delta):
        self.__delta = delta

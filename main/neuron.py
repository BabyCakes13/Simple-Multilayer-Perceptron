class Neuron:
    def __init__(self, weights, activation_function):
        try:
            self.__bias = weights[0]
            self.__weights = weights[1:]
        except ArrayIndexOutOfBounds as e:
            print("Weights does not have enough elements to split it into bias and weights.")
            return

        self.__activation_function = activation_function

    def compute(self, input):
        # insert the input for the bias
        if len(input) != len(self.__weights):
            raise Exception("The number of input is different than the number of weights in {}".format(self))

        sum = 0
        for i, w in zip(input, self.__weights):
            sum += i * w

        sum += self.__bias

        output = self.__activation_function(sum)
        return output

    def get_weights(self):
        return self.__weights

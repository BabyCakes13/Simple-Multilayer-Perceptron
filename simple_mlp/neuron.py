"""Module which implements the Neuron class."""


class Neuron:
    """Class which represents a neuron."""

    def __init__(self, weights,
                 activation_function,
                 activation_function_derivative=None):
        """Construct the Neuron class.

        Takes as arguments the weights, activation functions and their
        derivatives. The bias is delivered as the first element of the weighs.
        """
        try:
            self.__bias = weights[0]
            self.__weights = weights[1:]
        except IndexError:
            raise Exception("Weights does not have enough elements to split it"
                            "into bias and weights.")

        self.__activation_function = activation_function
        self.__activation_function_derivative = activation_function_derivative

        self.__output = None
        self.__delta = None

    def forward_propagate(self, input):
        """
        Forward propagate.

        The function forward propagates through the neuronal network by
        constructing the sum between each input and its weight, adding the bias
        at the end. The result represents the output for this neuron which will
        be paseed to the next layer.
        """
        if len(input) != len(self.__weights):
            raise Exception("The number of input is different than the number"
                            "of weights in {}".format(self))

        sum = 0
        for i, w in zip(input, self.__weights):
            sum += i * w

        sum += self.__bias

        output = self.__activation_function(sum)
        self.__output = output

        return output

    def adjust(self, learning_rate, input_neurons_list):
        """
        Adjust the weights after backpropagating.

        After the backward propagation of error is done, the weights of all the
        neurons are adjusted according to the following formula:

        adjusted_weight = weight + learning_rate * self.__delta * output

        The bias will also be adjusted.
        """
        adjusted_weights = []

        if len(self.__weights) != len(input_neurons_list):
            raise Exception("Weights and input neurons list are different"
                            "{} vs {}. Cannot adjust weights."
                            .format(self.__weights, input_neurons_list))

        for weight, input_neuron in zip(self.__weights, input_neurons_list):
            output = input_neuron.get_output()
            adjusted_weight = weight + learning_rate * self.__delta * output
            adjusted_weights.append(adjusted_weight)

        self.__weights = adjusted_weights
        self.__bias = self.__bias + learning_rate * self.__delta

    def get_derived_output(self):
        """Return the derived output which will be used in generating delta."""
        result = self.__activation_function_derivative(self.__output)
        return result

    def display(self):
        """Display the bias and weights of the current neuron."""
        print("Neuron {} has bias {} and weights: {}."
              .format(self, self.__bias, self.__weights))

    def get_weights(self) -> list:
        """Return the weights of the neuron."""
        return self.__weights

    def get_output(self) -> float:
        """Return the output of the neuron."""
        return self.__output

    def get_delta(self) -> float:
        """Return the delta of the neuron."""
        return self.__delta

    def get_bias(self) -> float:
        """Rreturn the bias of the neuron."""
        return self.__bias

    def get_weight(self, index):
        """Return the weight at a specific index from the neuron."""
        try:
            return self.__weights[index]
        except IndexError:
            raise Exception("The requested weight at index {} cannot be"
                            "retrieved. Index does not exist.".format(index))

    def set_output(self, output):
        """Set the output of the neuron. Needed for testing."""
        self.__output = output

    def set_delta(self, delta):
        """Set the delta of the neuron. Needed for testing."""
        self.__delta = delta

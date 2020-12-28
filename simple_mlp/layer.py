from simple_mlp import neuron as n


class Layer:
    def __init__(self, weights, activation_functions, activation_function_derivatives=None):
        self.__weights = weights
        self.__activation_functions = activation_functions
        self.__activation_function_derivatives = activation_function_derivatives

        self.__neurons = self.generate_neurons()

    def forward_propagate(self, input):
        output = []

        for n in self.__neurons:
            output.append(n.forward_propagate(input))

        return output

    def backward_propagate_output(self, expected_outputs):
        if len(self.__neurons) != len(expected_outputs):
            raise Exception("The number of neurons in the layer versus the expected outputs are different."
            "Cannot backward Propagate.")

        for neuron, eo in zip(self.__neurons, expected_outputs):
            error = eo - neuron.get_output()
            delta = error * neuron.get_derived_output()
            neuron.set_delta(delta)

    def backward_propagate_hidden(self, before_layer):
        for i, neuron in enumerate(self.__neurons):
            before_layer_neurons = before_layer.get_neurons()
            error = 0

            for bfn in before_layer_neurons:
                bfn_error = bfn.get_delta() * bfn.get_weight(index=i)
                error += bfn_error

            delta = error * neuron.get_derived_output()

            neuron.set_delta(delta)

    def adjust(self, learning_rate, input_layer):
        for neuron in self.__neurons:
            neuron.adjust(learning_rate, input_layer.get_neurons())

    def squared_mean_error(self, expected_output):
        squared_mean_error = 0

        for neuron, eo in zip(self.__neurons, expected_output):
            error = eo - neuron.get_output()
            squared_mean_error += error * error

        squared_mean_error = squared_mean_error / len(self.__neurons)
        return squared_mean_error

    def generate_neurons(self):
        neurons = []
        neurons_count = len(self.__weights)

        self.check_generate_neurons_possible(neurons_count)

        # for forward propagation only we do not need the derivatives.
        if self.__activation_function_derivatives is None:
            self.__activation_function_derivatives = [None] * len(self.__activation_functions)

        for w, a, ad in zip(self.__weights, self.__activation_functions, self.__activation_function_derivatives):
            new_neuron = n.Neuron(w, a, ad)
            neurons.append(new_neuron)

        return neurons

    def check_generate_neurons_possible(self, neurons_count):
        try:
            iter(self.__activation_functions)
        except TypeError:
            print("The activation function passed to the layer is set as common to the neurons.")
            self.__activation_functions = [self.__activation_functions] * neurons_count

        try:
            iter(self.__activation_function_derivatives)
        except TypeError:
            print("The activation function derivatives passed to the layer is set as common to the neurons.")
            self.__activation_function_derivatives = [self.__activation_function_derivatives] * neurons_count

        # check that the values necessary for generating the neurons are of equal length.
        if not all(len(lst) == neurons_count for lst in [self.__activation_function_derivatives, self.__activation_functions, self.__weights]):
            raise Exception("Cannot generate neurons in layer. The number of weights, activation functions and its derivatives are different.")

    def display(self):
        print("Layer {} has {} neurons.".format(self, len(self.__neurons)))
        for neuron in self.__neurons:
            neuron.display()

    def get_neurons_count(self):
        return len(self.__neurons)

    def get_neurons(self):
        return self.__neurons

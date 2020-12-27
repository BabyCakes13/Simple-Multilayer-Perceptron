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

    def generate_neurons(self):
        neurons = []
        size = len(self.__weights)

        # for forward propagation only we do not need the derivatives.
        if self.__activation_function_derivatives is None:
            self.__activation_function_derivatives = [None] * len(self.__activation_functions)

        for w, a, ad in zip(self.__weights, self.__activation_functions, self.__activation_function_derivatives):
            new_neuron = n.Neuron(w, a, ad)
            neurons.append(new_neuron)

        return neurons

    def display(self):
        print("Layer {} has {} neurons.".format(self, len(self.__neurons)))
        for neuron in self.__neurons:
            neuron.display()

    def get_neurons_count(self):
        return len(self.__neurons)

    def get_neurons(self):
        return self.__neurons

from simple_mlp import layer as l


class Network:
    def __init__(self, weights, activation_functions, activation_functions_derivatives):
        self.__weights = weights
        self.__activation_functions = activation_functions
        self.__activation_functions_derivatives = activation_functions_derivatives

        self.__layers = self.generate_layers()

    def generate_layers(self):
        layers = []

        if self.__activation_functions_derivatives is None:
            self.__activation_functions_derivatives = [None] * len(self.__activation_functions)

        for w, a, af in zip(self.__weights, self.__activation_functions, self.__activation_functions_derivatives):
            layer = l.Layer(w, a, af)
            layers.append(layer)

        if len(layers) == 0:
            raise Exception("No layer has been created.")

        return layers

    def get_layers(self):
        return self.__layers

    def forward_propagate(self, input):
        outputs = []

        for layer in self.__layers:
            outputs = layer.forward_propagate(input)
            input = outputs

        return outputs

    def backwards_propagation(self, expected_outputs):
        before_layer = self.__layers[-1]
        print("Backpropagating last layer with {} neurons.".format(before_layer.get_neurons_count()))
        before_layer.backward_propagate_output(expected_outputs)

        for layer in reversed(self.__layers[:-1]):
            print("Backpropagating layer with {} neurons.".format(layer.get_neurons_count()))

            layer.backward_propagate_hidden(before_layer)
            before_layer = layer

    def adjust(self, learning_rate, input):
        input_neurons_weights = [[1]] * len(input)
        input_layer = l.Layer(input_neurons_weights, [None] * len(input), None)

        for n, i in zip(input_layer.get_neurons(), input):
            n.set_output(i)

        for layer in self.__layers:
            layer.adjust(learning_rate, input_layer)
            input_layer = layer

    def squared_mean_error(self, expected_output):
        output_layer = self.__layers[-1]
        return output_layer.squared_mean_error(expected_output)

    def display(self):
        for layer in self.__layers:
            layer.display()

"""Module which implements the neural network."""


from simple_mlp import layer as lay


class Network:
    """Class which holds the neural network structure and its functions."""

    def __init__(self, weights,
                 activation_functions,
                 activation_functions_derivatives):
        """
        Construct the networkand its layers.

        The ingredients for the neural network generation are passed as
        parameters to the cosntructor. These will be passed through the layer,
        to each neuron.
        """
        self.__weights = weights
        self.__activation_functions = activation_functions
        self.__activation_functions_derivatives = \
            activation_functions_derivatives

        self.__layers = self.generate_layers()

    def generate_layers(self):
        """
        Generate the network layers.

        Using the weights, activation functions and their derivatives, each
        layer of the network will be constructed based on the structure of the
        weights.
        """
        layers = []

        # In case we need only to forward propagate without tuning the network
        # with backpropagation, the activation function derivatives are not
        # necessary, thus, they can be None. However, in order not to change
        # the structure of the code too much for this case, instead of working
        # with a None item, we work with a None list, which keeps the logic
        # of the implementation based on lists intact.
        if self.__activation_functions_derivatives is None:
            self.__activation_functions_derivatives = \
                [None] * len(self.__activation_functions)

        self.__prepare_layers_generation(len(self.__weights))

        for w, a, af in zip(self.__weights,
                            self.__activation_functions,
                            self.__activation_functions_derivatives):
            layer = lay.Layer(w, a, af)
            layers.append(layer)

        if len(layers) == 0:
            raise Exception("No layer has been created.")

        return layers

    def __prepare_layers_generation(self, layers_count):
        """
        Prepare the generation of network layers.

        This is the last point check before generating the layers, Thus, in
        case any requirement is not met, an exception is raised.
        """
        layer_ingredients = [self.__weights,
                             self.__activation_functions,
                             self.__activation_functions_derivatives]
        # pythonic check whether all the layer ingredients have the same length
        if not all(len(lst) == layers_count for lst in layer_ingredients):
            raise Exception("Cannot generate layers in network."
                            "The number of weights, activation functions and"
                            "their derivatives are different.")

    def forward_propagate(self, input):
        """
        Forward propagate through the layers.

        The forward propagate command is passed to each layer and each neuron.
        The logic is implemented at neuron level.
        """
        outputs = []

        for layer in self.__layers:
            outputs = layer.forward_propagate(input)
            input = outputs

        return outputs

    def backwards_propagation(self, expected_outputs):
        """
        Backpropagate through the network.

        The backpropagation is split into the case of the output and expected,
        and between two layers of the network. The backpropagate logic is
        implemented at neuron level.
        """
        previous_layer = self.__layers[-1]
        previous_layer.backward_propagate_output(expected_outputs)

        for layer in reversed(self.__layers[:-1]):
            layer.backward_propagate_hidden(previous_layer)
            previous_layer = layer

    def adjust(self, learning_rate, input):
        """
        Adjust weights for each layer after backpropagation.

        The adjust logic is made at neuron level. At this level, we are
        passing the command to each layer, and each layer will pass it to each
        neuron.
        """
        input_neurons_weights = [[1]] * len(input)
        # since we only need to adjust the weights, we do not care about the
        # activation function at this step.Therefore we can pass a None array
        input_layer = lay.Layer(input_neurons_weights,
                                [None] * len(input), None)
        input_layer_neurons = input_layer.get_neurons()

        if len(input_layer_neurons) != len(input):
            raise Exception("Cannot adjust weights. The number of input layer"
                            "neurons is different than the number of inputs.")

        for n, i in zip(input_layer.get_neurons(), input):
            n.set_output(i)

        for layer in self.__layers:
            layer.adjust(learning_rate, input_layer)
            input_layer = layer

    def squared_mean_error(self, expected_output):
        """
        Calculate the squared mean error.

        The logic is implemented at neuron level.
        """
        output_layer = self.__layers[-1]
        return output_layer.squared_mean_error(expected_output)

    def display(self):
        """Display the network."""
        for layer in self.__layers:
            layer.display()

    def get_layers(self):
        """Return the layers for this network."""
        return self.__layers

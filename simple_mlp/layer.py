"""Module which implement the Layer class."""


from simple_mlp import neuron as n


class Layer:
    """Class which represents one layer of a neural network."""

    def __init__(self, weights,
                 activation_functions,
                 activation_function_derivatives=None):
        """Set up the object variables."""
        self.__activation_functions = activation_functions
        self.__activation_function_derivatives = \
            activation_function_derivatives
        self.__weights = weights
        self.__neurons = self.generate_neurons()

    def forward_propagate(self, input) -> list:
        """Forward propagate on the neurons of the current layer.

        Gather all the outputs generated by the forward propagation in an
        output list which will be used by calculating the delta error of the
        neurons on backpropagating.
        """
        output = []

        for neuron in self.__neurons:
            output.append(neuron.forward_propagate(input))

        return output

    def backward_propagate_output(self, expected_outputs):
        """Backpropagate the error between the output layer and expected.

        This is a special case because we calculate the error between the
        output layer and the expected results, not the error difference between
        one layer and another.
        """
        if len(self.__neurons) != len(expected_outputs):
            raise Exception("The number of neurons in the layer versus the"
                            "number ofexpected outputs are different."
                            "Cannot backpropagate.")

        for neuron, expected_output in zip(self.__neurons, expected_outputs):
            error = expected_output - neuron.get_output()
            delta = error * neuron.get_derived_output()
            neuron.set_delta(delta)

    def backward_propagate_hidden(self, previous_layer):
        """Backpropagate the delta error between two layers."""
        # Calculate the error generated by all previous neurons connected to
        # the ones from the previous layer. Generate their deltas based on the
        # weighted sum beween each previous neuron with its weight connected to
        # the current one. The sum of errors is multiplied with the derivative
        # of the activation function used to get the results.
        for i, neuron in enumerate(self.__neurons):
            previous_layer_neurons = previous_layer.get_neurons()
            error = 0

            for pln in previous_layer_neurons:
                pln_error = pln.get_delta() * pln.get_weight(index=i)
                error += pln_error

            delta = error * neuron.get_derived_output()

            neuron.set_delta(delta)

    def adjust(self, learning_rate, input_layer):
        """Adjust the weights of the neurons from the current layer.

        After getting the results from the backpropagation, adjust the weights
        for all the neurons in the current layer.
        """
        # The computation for weight change is done at neuron level.
        for neuron in self.__neurons:
            neuron.adjust(learning_rate, input_layer.get_neurons())

    def squared_mean_error(self, expected_output):
        """Compute the squared mean error between the output and expected.

        The squared mean error is calculated at the output layer between the
        results generated by the neuron network and the expected results. More
        info at:https://www.freecodecamp.org/news/
        machine-learning-mean-squared-error-regression-line-c7dde9a26b93/
        """
        squared_mean_error = 0

        for neuron, eo in zip(self.__neurons, expected_output):
            error = eo - neuron.get_output()
            squared_mean_error += error * error

        squared_mean_error = squared_mean_error / len(self.__neurons)
        return squared_mean_error

    def generate_neurons(self):
        """
        Generate neurons to populate the current layer.

        At the constructor level the weights, activation functions and their
        derivatives are passed in order to have the knowledge to construct
        the neurons needed at this layer.
        """
        neurons = []
        neurons_count = len(self.__weights)

        self.prepare_neurons_generation(neurons_count)

        # For forward propagation only the derivatives do not need to be
        # specified; but in order to have the code working without creating too
        # many special cases, an array of None is created and treated the same
        # as when we would need the derivatives.
        if self.__activation_function_derivatives is None:
            self.__activation_function_derivatives = [None] * \
                len(self.__activation_functions)

        for w, a, ad in zip(self.__weights,
                            self.__activation_functions,
                            self.__activation_function_derivatives):
            new_neuron = n.Neuron(w, a, ad)
            neurons.append(new_neuron)

        return neurons

    def prepare_neurons_generation(self, neurons_count) -> None:
        """Prepare the neurons generation for the current layer.

        The passed constructor parameters might be different in length, which
        means that they have not been correctly constructed. The weights,
        activation functions and their derivatives lists must have the same
        length, since all neurons will have one of each. If any of these
        exceptions occur, they will be handled before attempting to generate
        the neurons.
        """
        # We must also check if the passed parameter is iterable (a list) or
        # not. If not, then we know that the passed function is not a list,
        # rather a universal for each neuron. Again in order not to change the
        # code too much, a list of functions is created from the passed one and
        # treated as a normal list of "separate" activations.
        try:
            iter(self.__activation_functions)
        except TypeError:
            print("One universal activation function set for all neurons.")
            self.__activation_functions = [self.__activation_functions] * \
                neurons_count

        try:
            iter(self.__activation_function_derivatives)
        except TypeError:
            print("One universal activation function derivative"
                  "set for all neurons.")
            self.__activation_function_derivatives = \
                [self.__activation_function_derivatives] * \
                neurons_count

        neuron_ingredients = [self.__activation_function_derivatives,
                              self.__activation_functions,
                              self.__weights]

        # pythonic check hat all neuron ingredients have the same length
        if not all(len(lst) == neurons_count for lst in neuron_ingredients):
            raise Exception("Cannot generate neurons in layer."
                            "The number of weights, activation functions and"
                            "their derivatives are different at layer {}.") \
                            .format(self)

    def display(self) -> None:
        """Display information about the current layer."""
        print("Layer {} has {} neurons.".format(self, len(self.__neurons)))
        for neuron in self.__neurons:
            neuron.display()

    def get_neurons_count(self) -> int:
        """Return the number of neurons at the current layer."""
        return len(self.__neurons)

    def get_neurons(self) -> list:
        """Return the list of neurons at the current layer."""
        return self.__neurons

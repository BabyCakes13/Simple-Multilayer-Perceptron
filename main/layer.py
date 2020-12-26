from main import neuron as n


class Layer:
    def __init__(self, weights, activation_functions):
        self.__weights = weights
        self.__activation_functions = activation_functions

        self.__neurons = self.generate_neurons()

    def forward_propagate(self, input):
        output = []

        for n in self.__neurons:
            output.append(n.forward_propagate(input))

        return output

    def generate_neurons(self):
        neurons = []
        size = len(self.__weights)

        for w, a in zip(self.__weights, self.__activation_functions):
            new_neuron = n.Neuron(w, a)
            neurons.append(new_neuron)

        return neurons

    def get_neurons_count(self):
        return len(self.__neurons)

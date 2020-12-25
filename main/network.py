from main import layer as l


class Network:
    def __init__(self, weights, activation_functions):
        self.__weights = weights
        self.__activation_functions = activation_functions

        self.__layers = self.generate_layers()

    def generate_layers(self):
        layers = []

        for w, a in zip(self.__weights, self.__activation_functions):
            layer = l.Layer(w, a)
            layers.append(layer)

        if len(layers) == 0:
            raise Exception("No layer has been created.")

        return layers

    def compute_output(self, input):
        outputs = []

        for layer in self.__layers:
            outputs = layer.compute_output(input)
            input = outputs

        return outputs

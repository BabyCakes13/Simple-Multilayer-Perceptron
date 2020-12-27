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

    def get_layers(self):
        return self.__layers

    def forward_propagate(self, input):
        outputs = []

        for layer in self.__layers:
            outputs = layer.forward_propagate(input)
            input = outputs

        return outputs

    def adjust(self, learning_rate, inputs):
        first_layer = self.__layers[0]
        first_layer.adjust(learning_rate, inputs)

        for layer in self.__layers[1:]:
            pass

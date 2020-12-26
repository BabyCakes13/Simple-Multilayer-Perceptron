
def generate_one_array(n):
    array = [1] * n
    return array


def generate_one_matrix(weights_count, neuron_count):
    matrix = [generate_one_array(weights_count)] * neuron_count
    return matrix


def generate_one_neuron_weights(weights_per_neuron_count):
    neuron_weights = [1] * weights_per_neuron_count
    return neuron_weights


def generate_one_neuron_weights_with_bias(weights_per_neuron_count, bias):
    neuron_weights = [1] * weights_per_neuron_count
    neuron_weights.insert(0, bias)
    return neuron_weights


def generate_one_layer_weights(weights_per_neuron_count, neuron_count):
    layer_weights = [generate_one_neuron_weights_with_bias(weights_per_neuron_count, bias=1)] * neuron_count
    return layer_weights


def generate_one_network_weights(input_count, neuron_count_per_layer_list):
    network_weights = []
    for nc in neuron_count_per_layer_list:
        network_weights.append(generate_one_layer_weights(weights_per_neuron_count=input_count,
                                                                      neuron_count=nc))
        input_count = nc

    return network_weights


def activation_function(x):
    return max(0, x)


def generate_layer_activation_functions(neuron_count):
    activation_functions = [activation_function] * neuron_count
    return activation_functions


def generate_network_activation_functions(network_layout):
    netwrok_activation_functions = []

    for neuron_count in network_layout:
        netwrok_activation_functions.append(generate_layer_activation_functions(neuron_count))

    return netwrok_activation_functions


def generate_one_and_minus_one_array(size):
    array = [1, -1] * (size // 2)
    return array

import functions


def calculate_neuron_value(inputs, weights, bias):
    inputs_len = len(inputs)
    weights_len = len(weights)
    ############################################################
    # error checking
    # Makes sure both values are the same length
    if inputs_len != weights_len:
        raise Exception('Length of both iterables must be the same')
    ############################################################
    value = 0
    for index in range(inputs_len):
        value += inputs[index] * weights[index]
    return functions.sigmoid(value + bias)
from micrograd.neural_network import MLP


def test_multilayer_perceptron():
    """
    Network Topology:   3 - 4 - 4 - 1
    Input:              2.0, 3.0, -1.0
    """

    x = [2.0, 3.0, -1.0]
    network = MLP(3, [4, 4, 1])
    output = network(x)
    assert (
        output.data <= 1 and output.data >= -1
    ), "Output should be in the range of tanh activation function [-1, 1]"

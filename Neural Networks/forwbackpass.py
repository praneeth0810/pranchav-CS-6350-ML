import numpy as np
from sgd import NeuralNetwork  


forward_pass_result_by_hand = np.array([[-2.437]])
forward_pass_neurons_by_hand = (
    np.array([[-6, 6]]), 
    np.array([[0.00247, 0.9975]]), 
    np.array([[-4, 4]]), 
    np.array([[0.01803, 0.9820]])
)

backward_propagation_by_hand = (
    np.array([[0.00101, 0.00153], [0.00101, 0.00153]]), 
    np.array([0.00101, 0.00153]), 
    np.array([[-0.0003, 0.00022], [-0.121, 0.0916]]), 
    np.array([-0.121, 0.0916]), 
    np.array([[-0.0618], [-3.3746]]), 
    np.array([-3.4368])
)


neural_network_3 = NeuralNetwork(width=2)

X_input = np.array([[1, 1]]) 
y_output = np.array([1])

neural_network_3.params = {
    "W1": np.array([[-2, 2], [-3, 3]]),
    "b1": np.array([-1, 1]),
    "W2": np.array([[-2, 2], [-3, 3]]),
    "b2": np.array([-1, 1]),
    "W3": np.array([[2], [-1.5]]),
    "b3": np.array([-1])
}

forward_pass_result, forward_pass_values = neural_network_3.forward_pass(X_input)

backward_pass_values = neural_network_3.backward_propagation(
    X_input, y_output, forward_pass_result, forward_pass_values
)

print('Forward Pass - By Hand vs Computed:')
print('Score (by hand):', forward_pass_result_by_hand)
print('Score (computed):', forward_pass_result)
print('Neuron Layer 1 (by hand):', forward_pass_neurons_by_hand[1])
print('Neuron Layer 1 (computed):', forward_pass_values[1])
print('Neuron Layer 2 (by hand):', forward_pass_neurons_by_hand[3])
print('Neuron Layer 2 (computed):', forward_pass_values[3])

print('\nBackward Pass - By Hand vs Computed:')
print('d Weight Vector 1 (by hand):', backward_propagation_by_hand[0])
print('d Weight Vector 1 (computed):', backward_pass_values["dW1"])
print('d Bias 1 (by hand):', backward_propagation_by_hand[1])
print('d Bias 1 (computed):', backward_pass_values["db1"])
print('d Weight Vector 2 (by hand):', backward_propagation_by_hand[2])
print('d Weight Vector 2 (computed):', backward_pass_values["dW2"])
print('d Bias 2 (by hand):', backward_propagation_by_hand[3])
print('d Bias 2 (computed):', backward_pass_values["db2"])
print('d Weight Vector 3 (by hand):', backward_propagation_by_hand[4])
print('d Weight Vector 3 (computed):', backward_pass_values["dW3"])
print('d Bias 3 (by hand):', backward_propagation_by_hand[5])
print('d Bias 3 (computed):', backward_pass_values["db3"])

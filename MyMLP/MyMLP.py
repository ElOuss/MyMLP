import numpy as np
from typing import List


class MyMLP:
    def __init__(self, npl: List[int]):
        self.struct = npl
        self.layers = len(npl) - 1
        self.weights = []

        # Initialize weights of the model between -1 and 1 (except for unused weights set to 0)
        for layer in range(self.layers + 1):
            self.weights.append([])

            if layer == 0:
                continue

            for prev_neurons in range(npl[layer - 1] + 1):
                self.weights[layer].append([])
                for curr_neurons in range(npl[layer] + 1):
                    self.weights[layer][prev_neurons].append(
                        0.0 if curr_neurons == 0 else np.random.uniform(-1.0, 1.0))


        # Create memory space to store neuron output values
        self.X = []
        for layer in range(self.layers + 1):
            self.X.append([])
            for neuron in range(npl[layer] + 1):
                self.X[layer].append(1.0 if neuron == 0 else 0.0)

        # Creating memory space to store semi-gradients associated with each neuron
        self.semi_gradients = []
        for layer in range(self.num_layers + 1):
            self.semi_gradients.append([])
            for neuron_idx in range(npl[layer] + 1):
                # Initializing the semi-gradients as 0.0
                self.semi_gradients[layer].append(0.0)

    # Propagation and updating of neuron output values from example inputs
    def _propagate(self, inputs: List[float], is_classification: bool):
        # Copying the input values into the input layer of the model
        for neuron_idx in range(self.npl[0]):
            self.X[0][neuron_idx + 1] = inputs[neuron_idx]

        # Recursively updating neuron output values, layer by layer
        for layer in range(1, self.layers + 1):
            for neuron_idx in range(1, self.npl[layer] + 1):
                total = 0.0
                for prev_neuron_idx in range(0, self.neurons_per_layer[layer - 1] + 1):
                    # Summing the weighted inputs from the previous layer
                    total += self.weights[layer][prev_neuron_idx][neuron_idx] * self.X[layer - 1][
                        prev_neuron_idx]

                # Applying the hyperbolic tangent (tanh) activation function if not in the last layer or if it's a classification task
                if layer < self.layers or is_classification:
                    total = np.tanh(total)

                # Updating the neuron's output value
                self.X[layer][neuron_idx] = total

    # Method for querying the model (inference)
    def predict(self, inputs: List[float], is_classification: bool):
        self._propagate(inputs, is_classification)
        return self.X[self.layers][1:]

    # Method for training the model from a labeled dataset
    def train(self, all_samples_inputs: List[List[float]], all_samples_expected_outputs: List[List[float]],
              is_classification: bool, iteration_count: int, alpha: float):
        # For a certain number of iterations
        for it in range(iteration_count):
            # Choosing a labeled example at random from the dataset
            k = np.random.randint(0, len(all_samples_inputs))
            inputs_k = all_samples_inputs[k]
            y_k = all_samples_expected_outputs[k]

            # Updating the neuron output values of the model from the inputs of the selected example
            self._propagate(inputs_k, is_classification)

            # Calculating the semi-gradients of neurons in the last layer
            for neuron_idx in range(1, self.neurons_per_layer[self.num_layers] + 1):
                self.semi_gradients[self.num_layers][neuron_idx] = (
                            self.neuron_outputs[self.num_layers][neuron_idx] - y_k[neuron_idx - 1])
                if is_classification:
                    self.semi_gradients[self.num_layers][neuron_idx] *= (
                                1 - self.neuron_outputs[self.num_layers][neuron_idx] ** 2)

            # Calculating the semi-gradients of neurons in previous layers recursively
            for layer in reversed(range(1, self.num_layers + 1)):
                for prev_neuron_idx in range(1, self.neurons_per_layer[layer - 1] + 1):
                    total = 0.0
                    for neuron_idx in range(1, self.neurons_per_layer[layer] + 1):
                        # Summing the weighted semi-gradients from the next layer
                        total += self.weights[layer][prev_neuron_idx][neuron_idx] * self.semi_gradients[layer][
                            neuron_idx]
                    # Updating the semi-gradient of the current neuron
                    self.semi_gradients[layer - 1][prev_neuron_idx] = (1 - self.neuron_outputs[layer - 1][
                        prev_neuron_idx] ** 2) * total

            # Updating the model weights
            for layer in range(1, self.num_layers + 1):
                for prev_neuron_idx in range(0, self.neurons_per_layer[layer - 1] + 1):
                    for neuron_idx in range(1, self.neurons_per_layer[layer] + 1):
                        # Updating the weight using gradient descent
                        self.weights[layer][prev_neuron_idx][neuron_idx] -= alpha * self.neuron_outputs[layer - 1][
                            prev_neuron_idx] * self.semi_gradients[layer][neuron_idx]

# Test

# Créer une instance de MyMLP avec la structure [2, 3, 1]
mlp = MyMLP([2, 3, 1])

# Afficher les poids initialisés
for layer_idx, layer_weights in enumerate(mlp.weights):
    print(f"Poids pour la couche {layer_idx}:")
    for prev_neurons, neuron_weights in enumerate(layer_weights):
        print(f"  Neurone {prev_neurons}:", neuron_weights)

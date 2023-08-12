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

# Test

# Créer une instance de MyMLP avec la structure [2, 3, 1]
mlp = MyMLP([2, 3, 1])

# Afficher les poids initialisés
for layer_idx, layer_weights in enumerate(mlp.weights):
    print(f"Poids pour la couche {layer_idx}:")
    for prev_neurons, neuron_weights in enumerate(layer_weights):
        print(f"  Neurone {prev_neurons}:", neuron_weights)

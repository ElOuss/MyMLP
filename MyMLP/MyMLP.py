import numpy as np
from typing import List
import os
import matplotlib.pyplot as plt
import json

class MyMLP:
    def __init__(self, npl: List[int]):
        self.struct = npl
        self.layers = len(npl) - 1
        self.weights = []
        self.output_value = []
        self.loss =[]

        # Initialisation des poids du modèle entre -1 et 1 (sauf pour les poids inutilisés fixés à 0)
        for layer in range(self.layers + 1):
            self.weights.append([])

            if layer == 0:
                continue

            for prev_neurons in range(npl[layer - 1] + 1):
                self.weights[layer].append([])
                for curr_neuron in range(npl[layer] + 1):
                    self.weights[layer][prev_neurons].append(
                        0.0 if curr_neuron == 0 else np.random.uniform(-1.0, 1.0))

        # Création d'un espace mémoire pour stocker les valeurs de sortie des neurones
        for layer in range(self.layers + 1):
            self.output_value.append([])
            for neuron in range(npl[layer] + 1):
                self.output_value[layer].append(1.0 if neuron == 0 else 0.0)

        # Création d'un espace mémoire pour stocker les semi-gradients associés à chaque neurone
        self.deltas = []
        for layer in range(self.layers + 1):
            self.deltas.append([])
            for neuron in range(npl[layer] + 1):
                # Initializing the semi-gradients as 0.0
                self.deltas[layer].append(0.0)

    # Propagation et mise à jour des valeurs de sortie des neurones à partir des entrées
    def _propagate(self, inputs: List[float], is_classification: bool):
        # Copie des valeurs d'entrée dans la couche d'entrée du modèle
        for neuron in range(self.struct[0]):
            self.output_value[0][neuron + 1] = inputs[neuron]

        # Mise à jour récursive des valeurs de sortie des neurones, couche par couche
        for layer in range(1, self.layers + 1):
            for neuron in range(1, self.struct[layer] + 1):
                total = 0.0
                for prev_neuron in range(0, self.struct[layer - 1] + 1):
                    # Somme des entrées pondérées de la couche précédente
                    total += self.weights[layer][prev_neuron][neuron] * self.output_value[layer - 1][
                        prev_neuron]

                # Application de la fonction d'activation (tanh) si ce n'est pas la dernière couche ou si c'est une tâche de classification
                if layer < self.layers or is_classification:
                    total = np.tanh(total)

                # Mise à jour de la valeur de sortie du neurone
                self.output_value[layer][neuron] = total

    # Méthode pour interroger le modèle
    def predict(self, inputs: List[float], is_classification: bool):
        self._propagate(inputs, is_classification)
        return self.output_value[self.layers][1:]

    # Méthode pour entraîner le modèle à partir d'un ensemble de données étiquetées
    def train(self, all_samples_inputs: List[List[float]], all_samples_expected_outputs: List[List[float]],
              is_classification: bool, iteration_count: int, alpha: float):
        # Pour un certain nombre d'itérations
        for it in range(iteration_count):
            # Choix d'un exemple étiqueté au hasard dans l'ensemble de données
            k = np.random.randint(0, len(all_samples_inputs))
            inputs_k = all_samples_inputs[k]
            y_k = all_samples_expected_outputs[k]

            # Mise à jour des valeurs de sortie des neurones du modèle à partir des entrées de l'exemple sélectionné
            self._propagate(inputs_k, is_classification)

            # Calcul des semi-gradients des neurones dans la couche
            for neuron_idx in range(1, self.struct[self.layers] + 1):
                self.deltas[self.layers][neuron_idx] = (
                        self.output_value[self.layers][neuron_idx] - y_k[neuron_idx - 1])
                if is_classification:
                    self.deltas[self.layers][neuron_idx] *= (
                            1 - self.output_value[self.layers][neuron_idx] ** 2)

            # Calcul des semi-gradients des neurones dans les couches précédentes de manière récursive
            for layer in reversed(range(1, self.layers + 1)):
                for prev_neuron_idx in range(1, self.struct[layer - 1] + 1):
                    total = 0.0
                    for neuron_idx in range(1, self.struct[layer] + 1):
                        # Somme des semi-gradients pondérés de la couche suivante
                        total += self.weights[layer][prev_neuron_idx][neuron_idx] * self.deltas[layer][
                            neuron_idx]
                    # Mise à jour du semi-gradient du neurone actuel
                    self.deltas[layer - 1][prev_neuron_idx] = (1 - self.output_value[layer - 1][
                        prev_neuron_idx] ** 2) * total

            # Updating the model weights
            for layer in range(1, self.layers + 1):
                for prev_neuron_idx in range(0, self.struct[layer - 1] + 1):
                    for neuron_idx in range(1, self.struct[layer] + 1):
                        # Updating the weight using gradient descent
                        self.weights[layer][prev_neuron_idx][neuron_idx] -= alpha * self.output_value[layer - 1][
                            prev_neuron_idx] * self.deltas[layer][neuron_idx]
            if it%1000 ==0.0:
                total_loss = 0
                for inputs, expected_outputs in zip(all_samples_inputs, all_samples_expected_outputs):
                    self._propagate(inputs, is_classification = True)
                    outputs = self.output_value[self.layers][1:]
                    example_loss =self.loss_mse(outputs, expected_outputs)
                    total_loss += example_loss

                average_loss = total_loss / len(all_samples_inputs)
                self.loss.append(average_loss)


    def loss_entropy(self, y_pred, y_true):
        # Calcule la différence entre les valeurs prédites et les valeurs cibles réelles.
        difference = y_pred - y_true
        # Convertir les valeurs cibles réelles et prédites en NumPy arrays.
        y_true_float = np.array(y_true)
        y_pred_float = np.array(y_pred)

        # Calcule la fonction de perte de cross-entropy.
        loss = -np.sum(
            y_true_float * np.log(y_pred_float + 1e-10) + (1 - y_true_float) * np.log(1 - y_pred_float + 1e-10))

        return loss

    def loss_mse(self,outputs, expected_outputs):
        loss = 0.0

        for output, expected in zip(outputs, expected_outputs):
            loss += (output - expected) ** 2

        return loss / (2.0 * len(outputs))


    def calculate_accuracy(self, all_samples_inputs: List[List[float]],
                           all_samples_expected_outputs: List[List[float]]):
        correct_predictions = 0
        num_samples = len(all_samples_inputs)

        for i in range(num_samples):
            predicted_outputs = np.array(self.predict(all_samples_inputs[i], is_classification=True))
            expected_outputs_i = all_samples_expected_outputs[i]

            # Convert predicted outputs to binary values (-1 or 1) based on threshold
            binary_predicted = np.sign(predicted_outputs)
            binary_expected = np.sign(expected_outputs_i)

            if np.array_equal(binary_predicted, binary_expected):
                correct_predictions += 1

        accuracy = correct_predictions / num_samples
        return accuracy

    def calculate_accuracy_multi_class(self, all_samples_inputs: List[List[float]],
                           all_samples_expected_outputs: List[List[float]]):
        correct_predictions = 0
        num_samples = len(all_samples_inputs)

        for i in range(num_samples):
            predicted_outputs = np.array(self.predict(all_samples_inputs[i], is_classification=True))
            expected_outputs_i = all_samples_expected_outputs[i]

            # convertir les valeur predites en labels (0, 1, 2) argmax
            predicted_class = np.argmax(predicted_outputs)
            expected_class = np.argmax(expected_outputs_i)

            if predicted_class == expected_class:
                correct_predictions += 1

        accuracy = correct_predictions / num_samples
        return accuracy
    def load_images(dataset_path):
        images = []
        labels = []

        # Parcours les classes d'images
        for class_name in os.listdir(dataset_path):
            class_path = os.path.join(dataset_path, class_name)
            if not os.path.isdir(class_path):
                continue  # Ignore les fichiers qui ne sont pas des répertoires

            for image_name in os.listdir(class_path):
                image_path = os.path.join(class_path, image_name)
                # Charge les images dans des numpy arrays
                image = np.array(plt.imread(image_path))
                images.append(image)
                labels.append(class_name)

        return images, labels

    def preprocess_images(images, scale):
        #Prétraite les images
        images = [image / scale for image in images]
        return images

    def split_dataset(images, labels, train_ratio=0.8):
        #Découpe le dataset en train set et test set
        n_images = len(images)
        n_train_images = int(n_images * train_ratio)
        n_test_images = n_images - n_train_images
        train_images = images[:n_train_images]
        train_labels = labels[:n_train_images]
        test_images = images[n_train_images:]
        test_labels = labels[n_train_images:]
        return train_images, train_labels, test_images, test_labels

    # Fonction pour sauvegarder le modèle en JSON
    def save_model_to_json(model, filename):
        # Crée un dictionnaire qui contient la structure du modèle et ses poids
        model_data = {
            "struct": model.struct,
            "weights": model.weights,
        }
        # Ouvre le fichier JSON en mode write
        with open(filename, "w") as json_file:
            # Écrit les données du modèle dans le fichier JSON
            json.dump(model_data, json_file)

    # Fonction pour charger un modèle depuis un fichier JSON
    def load_model_from_json(filename):
        # Ouvre le fichier JSON en mode (read) et l'assigner à la variable json_file.
        with open(filename, "r") as json_file:
            model_data = json.load(json_file)

        # Crée une instance de la classe MyMLP avec la structure du modèle dans model_data.
        loaded_model = MyMLP(model_data["struct"])

        loaded_model.weights = model_data["weights"]

        # Retourne le modèle chargé depuis le fichier JSON.
        return loaded_model

    def flatten_images(images):
        flattened_images = []
        for image in images:
            # Aplatit chaque image
            flattened_image = image.reshape(-1)
            flattened_images.append(flattened_image)
        return flattened_images

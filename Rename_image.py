import os

# Liste des classes
classes = ["Barbe à papa", "Churros", "Pomme damour"]

# Fonction pour renommer une image
def rename_image(image_path, class_name, index):
    # Obtenir le nom original de l'image
    image_name = os.path.basename(image_path)

    # Renommer l'image
    new_image_name = f"{class_name}_image{index:03d}"
    os.rename(image_path, os.path.join(os.path.dirname(image_path), new_image_name))


# Itérer sur les classes
for class_name in classes:
    # Obtenir le dossier de la classe
    class_path = os.path.join("/Users/ousselhabachi/Desktop/MyMLP/data", class_name)


    # Obtenir la liste des images dans le dossier
    image_paths = os.listdir(class_path)

    # Itérer sur les images
    for index, image_path in enumerate(image_paths):
        # Renommer l'image
        rename_image(image_path, class_name, index)

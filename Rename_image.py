import os

# Liste des classes
classes = ["Barbe a Papa", "Churros", "Pomme damour"]

# Fonction pour renommer une image
def rename_image(image_path, class_name, index):
    image_name = os.path.basename(image_path)

    # Renommer l'image
    new_image_name = f"{class_name}_image{index:03d}"
    os.rename(image_path, os.path.join(os.path.dirname(image_path), new_image_name))

data_dir = "data"

# Itérer sur les classes
for class_name in classes:
    class_path = os.path.join(data_dir, class_name)

    image_names = os.listdir(class_path)

    for index, image_name in enumerate(image_names):
        image_path = os.path.join(class_path, image_name)

        rename_image(image_path, class_name, index)

print("Images renommées avec succès.")

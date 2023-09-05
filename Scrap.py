import os
import requests
from bs4 import BeautifulSoup

# Lien de la page Google Images pour "Barbe à papa"
link = 'https://www.google.com/search?q=churros+chocolat+chaud&tbm=isch&hl=fr&chips=q:churros+chocolat+chaud,online_chips:churros+espagnols:jP46ruH5Ibk%3D&sa=X&ved=2ahUKEwjE14fzy5GBAxWBrUwKHTRMCSIQ4lYoA3oECAEQOg&biw=1440&bih=669'
# Nom de la classe d'images à récupérer
class_name = 'Churros'

# Répertoire où les images seront sauvegardées
save_dir = os.path.join('data', class_name)

# Fonction pour récupérer les images
def get_images(link, class_name, save_dir):

    # Obtenir le contenu HTML de la page Google Images
    r = requests.get(link)
    soup = BeautifulSoup(r.content, 'html.parser')

    # Trouver toutes les balises 'img'
    img_tags = soup.find_all('img')

    # Vérifier si le répertoire de sauvegarde existe, sinon le créer
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Boucler sur toutes les balises 'img' pour télécharger les images
    for index, img in enumerate(img_tags):

        # Obtenir l'URL de l'image
        image_url = img['src']

        # Vérifier si l'URL est valide (commence par "http://" ou "https://")
        if not image_url or not (image_url.startswith("http://") or image_url.startswith("https://")):
            continue

        # Télécharger l'image et la sauvegarder dans le répertoire approprié avec un nom unique
        image_data = requests.get(image_url).content
        with open(os.path.join(save_dir, f"{class_name}_{index + 50}.jpg"), 'wb') as f:
            f.write(image_data)




get_images(link, class_name, save_dir)

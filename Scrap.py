import os
import requests
from bs4 import BeautifulSoup

# Lien de la page Google Images pour "Barbe à papa"
link = 'https://www.google.com/search?sxsrf=AB5stBjczR_iF1o495RIpyc2D9V3D3b0VQ:1691252190858&q=barbe+%C3%A0+papa&tbm=isch&source=lnms&sa=X&ved=2ahUKEwiJ476w9cWAAxX8dqQEHWYaAa0Q0pQJegQIDBAB&biw=1440&bih=726&dpr=2'

# Nom de la classe d'images à récupérer
class_name = 'Barbe à papa'

# Répertoire où les images seront sauvegardées
save_dir = 'Data'

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
        with open(os.path.join(save_dir, f"{class_name}_{index}.jpg"), 'wb') as f:
            f.write(image_data)




get_images(link, class_name, save_dir)

import os
import requests
from bs4 import BeautifulSoup


link = 'https://www.google.com/search?sxsrf=AB5stBjczR_iF1o495RIpyc2D9V3D3b0VQ:1691252190858&q=barbe+%C3%A0+papa&tbm=isch&source=lnms&sa=X&ved=2ahUKEwiJ476w9cWAAxX8dqQEHWYaAa0Q0pQJegQIDBAB&biw=1440&bih=726&dpr=2'
class_name = 'barbe à papa'
save_dir = 'Data'

def get_images(link, class_name, save_dir):
    r = requests.get(link)
    soup = BeautifulSoup(r.content, 'html.parser')

    img_tags = soup.find_all('img')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for index, img in enumerate(img_tags):
        image_url = img['src']
        image_data = requests.get(image_url).content

        with open(os.path.join(save_dir, f"{class_name}_{index}.jpg"), 'wb') as f:
            f.write(image_data)




get_images(link, class_name, save_dir)

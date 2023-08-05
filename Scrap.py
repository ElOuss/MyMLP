import requests
from bs4 import BeautifulSoup


url = "https://www.google.com/search?q=cotton+candy+photoshoot&tbm=isch&ved=2ahUKEwj657uR6sWAAxWInCcCHZHKAZ8Q2-cCegQIABAA&oq=cotton+candy+photosh&gs_lcp=CgNpbWcQARgAMgcIABATEIAEMgcIABATEIAEMggIABAFEB4QEzIICAAQCBAeEBM6BAgjECc6CAgAEAcQHhATOgUIABCABDoECAAQHlDjBljYLGCkM2gAcAB4AIABRIgBogaSAQIxM5gBAKABAaoBC2d3cy13aXotaW1nwAEB&sclient=img&ei=FWrOZLoJiLmewQ-RlYf4CQ&bih=726&biw=1440"
def getdata(link):
    r = requests.get(link)
    return r.text

htmldata = getdata(url)
soup = BeautifulSoup(htmldata, 'html.parser')


for item in soup.findAll('img'):
    print(item['src'])
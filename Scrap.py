import requests
from bs4 import BeautifulSoup


url = "https://www.google.com/search?sxsrf=AB5stBiCsky1e-hJDKTRHgS31usKRxe1Pw:1691240841574&q=candy+floss&tbm=isch&source=lnms&sa=X&ved=2ahUKEwi5ut2My8WAAxXjUaQEHY_ID0wQ0pQJegQIDxAB&biw=1440&bih=726&dpr=2"

def getdata(lien):
    r = requests.get(lien)
    return r.text

htmldata = getdata(url)
soup = BeautifulSoup(htmldata, 'html.praser')


for item in soup.findAll('img'):
    print(item['src'])
import json
import requests
from bs4 import BeautifulSoup

def scrap():
    resp = requests.get("http://www.netlingo.com/acronyms.php")
    soup = BeautifulSoup(resp.text, "html.parser")
    slangdict = {}
    for div in soup.findAll('div', attrs={'class': 'list_box3'}):
        for li in div.findAll('li'):
            for a in li.findAll('a'):
                key = a.text
                value = li.text.split(key)[1]
                slangdict[key] = value

    with open('/home/ivan/Documents/git_repos/Sentiment-Analysis-on-Twitter/data/myslang.json', 'w') as f:
        json.dump(slangdict, f, indent=2)


if __name__ == '__main__':
    scrap()

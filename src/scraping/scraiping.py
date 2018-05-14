import urllib
from bs4 import BeautifulSoup
import pandas as pd
from time import sleep

url_base = 'https://okwave.jp/qa/q%ID%.html'

num_items = 10000
i = 0
titles = []
descs =  []
anses = []
for id in range(9498001, 1, -1):
    url = url_base.replace("%ID%", str(id))
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as res:
            body = res.read()
        body = body.decode('utf-8')
        soup = BeautifulSoup(body, "html.parser")
        title = soup.find("title").text
        desc = soup.find("div", class_="q_desc").text
        ans = soup.find('div', class_='a_textarea').text
    except:
        continue
    titles += [title]
    descs += [desc]
    anses += [ans]
    print(title)
    print(desc)
    print(ans)
    print()
    i += 1
    print("{}/{}".format(i, num_items))
    if i > num_items:
        break
    sleep(0.1)

df = pd.DataFrame()
df['title'] = titles
df['question'] = descs
df['answer'] = anses

df.to_csv("okwave.csv")

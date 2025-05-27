html_content = """
<html>
<title>Data Science is Fun</title>

<body>
    <h1>Data Science is Fun</h1>
    <div id='paragraphs' class='text'>
        <p id='paragraph 0'>Paragraph 0
            Paragraph 0 Paragraph 0 Paragraph 0 Paragraph 0 Paragraph 0 Paragraph 0
            Paragraph 0 Paragraph 0 Paragraph 0 Paragraph 0 Paragraph 0 Paragraph 0
            Paragraph 0 Paragraph 0 Paragraph 0 Paragraph 0 Paragraph 0 Paragraph 0
            Paragraph 0 Paragraph 0 Paragraph 0 Paragraph 0 Paragraph 0 Paragraph 0
            Paragraph 0 Paragraph 0 Paragraph 0 Paragraph 0 Paragraph 0 Paragraph 0
            Paragraph 0 Paragraph 0 Paragraph 0 Paragraph 0 Paragraph 0 Paragraph 0
            Paragraph 0 Paragraph 0 Paragraph 0 </p>
        <p id='paragraph 1'>Paragraph 1
            Paragraph 1 Paragraph 1 Paragraph 1 Paragraph 1 Paragraph 1 Paragraph 1
            Paragraph 1 Paragraph 1 Paragraph 1 Paragraph 1 Paragraph 1 Paragraph 1
            Paragraph 1 Paragraph 1 Paragraph 1 Paragraph 1 Paragraph 1 Paragraph 1
            Paragraph 1 Paragraph 1 Paragraph 1 Paragraph 1 Paragraph 1 Paragraph 1
            Paragraph 1 Paragraph 1 Paragraph 1 Paragraph 1 Paragraph 1 Paragraph 1
            Paragraph 1 Paragraph 1 Paragraph 1 Paragraph 1 Paragraph 1 Paragraph 1
            Paragraph 1 Paragraph 1 Paragraph 1 </p>
        <p id='paragraph 2'>Here is a link to <a href='https://www.mail.ru'>Mail ru</a></p>
    </div>
    <div id='list' class='text'>
        <h2>Common Data Science Libraries</h2>
        <ul>
            <li>NumPy</li>
            <li>SciPy</li>
            <li>Pandas</li>
            <li>Scikit-Learn</li>
        </ul>
    </div>
    <div id='empty' class='empty'></div>
</body>

</html>
"""

from bs4 import BeautifulSoup as bs

soup = bs(html_content, "lxml")

# title = soup.find("title")
# print(title)
# print(type(title))
# print(title.text)

# print(soup.body.text)
# print(soup.body.p)

# pList = soup.body.find_all("p")
# for i, p in enumerate(pList):
#     print(p.text)
#     print("---------------")

# print([bullet.text for bullet in soup.body.find_all("li")])

# p2 = soup.find(id="paragraph 2")
# print(p2.text)

divAll = soup.find_all("div")
print(divAll)

divClassText = soup.find_all("div", class_="text")
print(divClassText)

for div in divClassText:
    print("---------------")
    id = div.get("id")
    print(id)
    print(div.text)
    print("---------------")

soup.body.find(id="paragraph 0").decompose()
soup.body.find(id="paragraph 1").decompose()

print(soup.body.find(id="paragraphs"))

new_p = soup.new_tag("p")
print(new_p)
print(type(new_p))

new_p.string = "Новое содержание"
print(new_p)

soup.find(id="empty").append(new_p)

print(soup)

# Возьмём пример с сайта
from urllib.request import urlopen

url = "https://ya.ru"
html_content = urlopen(url).read()
print(html_content[:1000])

sp = bs(html_content, "lxml")
print(sp.title.text)
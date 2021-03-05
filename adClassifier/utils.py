from html.parser import HTMLParser
from pathlib import Path
from tqdm import tqdm
import json
import pandas as pd
from io import StringIO
from collections import Counter

def make_combined_df(path: Path):
    files = path.glob("*.json")
    df = None
    for file in tqdm(sorted(files)):
        # print(file)
        state, age, politics, gender, page = file.stem.split("_")
        #file = Path("../louhos-analytics/db/us_vaalit/propublica/data/Alabama_20_any_male.json")
        try:
            with open(str(file), "r", encoding="utf-8") as f:
                data=json.load(f)["ads"]
                d = pd.DataFrame(data)
                d["state"] = state
                d["age"] = age
                d["politics"] = politics
                d["gender"] = gender

            if df is None:
                df = d


            else:
                df = df.append(d)
        except FileNotFoundError:
            print("File {} not found".format(file))

    return df



class ImgFromHTMLParser(HTMLParser):
    """
    Parser class for fetching images from ad HTML.
    """
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.text = StringIO()
        self.__text = ""
        self.urls = []

    def handle_starttag(self, tag, attrs):
        if tag == 'img':
            src = dict(attrs).get('src')
            if src:
                self.urls.append(src)

    def handle_data(self, data):
        # print("Data     :", data)
        self.text.write(data)

    def get_data(self):
        return self.text.getvalue()

    def text(self):
        return ''.join(self.__text).strip()


def get_img_from_html(x):
    parser = ImgFromHTMLParser()
    try:
        parser.feed(x["html"])
        actual_urls = [x for x in parser.urls if x.startswith("http")]
        # print(Counter(actual_urls))
        return Counter(actual_urls).most_common(1)[0][0]
    except TypeError:
        print("TypeError for \n{}".format(x))
        return None

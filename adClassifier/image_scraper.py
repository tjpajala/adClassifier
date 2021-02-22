import requests
from pathlib import Path
import itertools
import os
import time
from tqdm import tqdm
import json
import logging
import urllib.request
import sys
from html.parser import HTMLParser
from io import StringIO
from collections import Counter
import numpy as np
import pandas as pd
from urllib.error import HTTPError, URLError
sys.path.append(os.path.relpath("."))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


DATA_FOLDER = Path(__file__).parent / "data"


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


d = pd.read_csv(DATA_FOLDER / "en-US.csv.gz")
# fetching images
IMAGE_FOLDER = DATA_FOLDER / "img"
image_urls = []
im = d.images.apply(lambda x: x[1:].replace("}", "").split(",")[0])
for i in tqdm(range(len(d))):
    if len(im[i]) == 0:
        print("Images field does not exist in row {}, trying to parse from html.".format(i))
        try:
            image_urls.append([get_img_from_html(x) for x in d.iloc[i]["ads"]])
        except (KeyError, IndexError):
            print("Could not find image for row {} even from HTML.".format(i))
    else:
        image_urls.append(im[i])
    #logger.info(image_urls[0])
#image_urls = [item for sublist in image_urls for item in sublist]


image_urls = set(image_urls)
all_files = [x.name for x in IMAGE_FOLDER.glob("*")]
image_urls = [x for x in image_urls if "" + urllib.request.urlsplit(x).path.split("/")[-1] not in all_files]
logger.info("Fetching {} images".format(len(image_urls)))
fetch_counter = 1
for img in tqdm(image_urls):
    if fetch_counter % 100 == 0:
        logger.info(fetch_counter)
    filename = "" + urllib.request.urlsplit(img).path.split("/")[-1]
    # if filename exists, skip download
    if filename in all_files:
        logger.info("Found image {} already, skipping".format(filename))
        continue
    #logger.info("Fetching image {}, saving as {}".format(img, filename))
    try:
        urllib.request.urlretrieve(img, IMAGE_FOLDER / filename)
        fetch_counter = fetch_counter + 1
    except HTTPError:
        logger.info("HTTPError")
        continue
    except URLError:
        logger.info("URLError")
        continue

    if fetch_counter > 40000:
        logger.info("Stopping due to fetch counter!")
        break
    # logger.info("Sleeping for 1 second")
    #time.sleep(1)

print("Scraping process completed!")

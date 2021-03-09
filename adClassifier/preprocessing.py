import pandas as pd
import os
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import yaml
import json
from tqdm import tqdm
from collections import Counter
from sklearn.model_selection import train_test_split

sys.path.append(os.path.relpath("."))
from adClassifier.utils import get_img_from_html

params = yaml.safe_load(open('adClassifier/params.yaml'))['preprocessing']
DATA_FOLDER = Path(params["data_folder"])
COMPLETIONS_FOLDER = Path(params["completions_folder"])
test_percent_size = params['test_percent_size']

print(str(DATA_FOLDER.absolute()))

df = pd.read_csv(str(DATA_FOLDER / "en-US.csv.gz"))
# retain only ads that are likely to be political
df = df.loc[df.political_probability>0.95]

completions = {}
print("Looking in {}".format(COMPLETIONS_FOLDER))
json_files = list(COMPLETIONS_FOLDER.rglob("*.json"))
print("Found {} json files".format(len(json_files)))
print(json_files[0:2])

for jf in tqdm(json_files):
    with open(jf,"r") as f:
        d = json.load(f)
        c_id = str(d["data"]["id"])
        if len(d["completions"][0]["result"]) > 0:
            val = d["completions"][0]["result"][0]["value"]["choices"]
        else:
            print("Skipping {} with empty result".format(jf.name))
            continue
        #print("ID {}, choice {}".format(jf.name ,val))
        completions[c_id] = val[0]
print("Found total {} completions.".format(len(completions)))
print("Completion distribution: \n{}".format(Counter(completions.values())))


print("{} rows when beginning filtering.".format(len(df)))
ids_to_drop = [k for k in completions.keys() if completions[k] in ["Drop","Neutral"]]
print("Dropping {} obs as label is Drop or Neutral".format(len(ids_to_drop)))
df = df[~df.id.isin(ids_to_drop)]
print("{} rows after filtering.".format(len(df)))

print(df.shape)

df["created_date"] = df.created_at.apply(lambda x: datetime.strptime(x.split(" ")[0],"%Y-%m-%d"))
df.updated_at = df.updated_at.fillna(df.created_at)
df["updated_date"] = df.updated_at.apply(lambda x: datetime.strptime(x.split(" ")[0],"%Y-%m-%d"))
df["message"] = df.message.fillna("", inplace=False)
df["political"] = df.political.fillna(-1)
df["not_political"] = df.not_political.fillna(-1)
df["title"].fillna("", inplace=True)
df["id"] = df.id.astype(str)
df["paid_for_by"].fillna("", inplace=True)
print("Fillna completed.")

# fill images from same id
#df["images"] = df.images.apply(lambda x: x if ((isinstance(x, list)) & (len(x)>0)) else "")
#df.loc[df.images.isnull(),"images"] = df.loc[df.images.isnull()].apply(lambda x: [])
df["images"] = [ [""] if x is np.NaN else x for x in df['images'] ]
df["images"] = [ [""] if x==[] else x for x in df['images'] ]
#print(df.loc[df.title.str.contains("Amanda Stuck"),["id","images"]])
#print(df.loc[:,["id","images"]])
df["images"] = df.images.map(lambda x: x[0])
most_common = df.groupby("id")["images"].agg(lambda x:x.value_counts().index[0]).reset_index()
#most_common["images"] = most_common.images.apply(lambda x: x)
df.loc[df.images == "","images"] = np.nan
# first try to fill from other items with same id
df.loc[df.images.isnull(),"images"] = df.loc[df.images.isnull(),"id"].map(most_common.set_index("id").images)
# if we still don't have the image, try to find it from HTML
df.loc[df.images.isnull(),"images"] = df.loc[df.images.isnull(),"html"].apply(lambda x: get_img_from_html(x))
# then fill from other items again
df.loc[df.images.isnull(),"images"] = df.loc[df.images.isnull(),"id"].map(most_common.set_index("id").images)
#print("Fixed")
#print(df.loc[df.title.str.contains("Amanda Stuck"),["id","images"]])
df["message_label"] = np.nan

df["message_label"] = df.id.apply(lambda x: np.nan if x not in completions.keys() else completions[x])
df["message_label"] = df.message_label.replace({"Republican":1, "Democrat":0})

print(df.message_label.value_counts(dropna=False))

df_test = df.loc[~df.message_label.isnull(),:]
df_train = df.loc[df.message_label.isnull(),:]
print(df.message_label.value_counts())

df_validation, df_test = train_test_split(df_test, test_size=test_percent_size)

print('Number of observations in the training data:', len(df_train))
print('Number of observations in the test data:', len(df_test))
print("Classes in train: \n{}".format(df_train.message_label.value_counts()))
print("Classes in test: \n{}".format(df_test.message_label.value_counts()))

df.to_parquet(DATA_FOLDER / "df.parquet.gzip", compression="gzip")
df_train.to_parquet(DATA_FOLDER / "df_train.parquet.gzip", compression="gzip")
df_test.to_parquet(DATA_FOLDER / "df_test.parquet.gzip", compression="gzip")
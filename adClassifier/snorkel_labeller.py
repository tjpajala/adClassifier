import pandas as pd
import os
import sys
import hvplot
import hvplot.pandas
from pathlib import Path
from datetime import datetime
import numpy as np
import yaml

from snorkel.labeling import labeling_function, PandasLFApplier, LFAnalysis, filter_unlabeled_dataframe
from snorkel.labeling.model import LabelModel
from snorkel.utils import probs_to_preds
from snorkel.preprocess import preprocessor

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split

sys.path.append(os.path.relpath(".."))
from adClassifier.utils import make_combined_df, DATA_FOLDER, ImgFromHTMLParser, get_img_from_html
from adClassifier.models import image_classifier, message_classifier

from mlflow import log_metric, log_param, log_artifacts



params = yaml.safe_load(open('adClassifier/params.yaml'))['labeling_functions']
DATA_FOLDER = Path(params["data_folder"])
print(DATA_FOLDER)
METRICS_FOLDER = Path(params["metrics_folder"])
SCORES_FILE = METRICS_FOLDER / "scores.json"
AUC_FILE = METRICS_FOLDER / "auc.json"
seed = params["seed"]
n_epochs = params["n_epochs"]
log_freq = params["log_freq"]
n_estimators = params["n_estimators"]
class_weight = params["class_weight"]



ABSTAIN = -1
DEM = 0
REP = 1


df_train = pd.read_parquet(DATA_FOLDER / "df_train.parquet.gzip")
df_test = pd.read_parquet(DATA_FOLDER / "df_test.parquet.gzip")

print('Number of observations in the training data:', len(df_train))
print('Number of observations in the test data:', len(df_test))
print("Classes in train: \n{}".format(df_train.message_label.value_counts()))
print("Classes in test: \n{}".format(df_test.message_label.value_counts()))


@labeling_function()
def lf_mentions_trump(x):
    # Return label of REP if Trump is mentioned
    return REP if "trump" in x.message.lower() else ABSTAIN

@labeling_function()
def lf_mentions_maga(x):
    # Return REP if maga is mentioned
    return REP if "maga" in x.message.lower() else ABSTAIN

@labeling_function()
def lf_mentions_bidenharris(x):
    # Return DEM if bidenharris is mentioned
    return DEM if "bidenharris" in x.message.lower() else ABSTAIN

@labeling_function()
def lf_mentions_biden(x):
    # Return DEM if bidenharris is mentioned
    return DEM if "biden" in x.message.lower() else ABSTAIN

@labeling_function()
def lf_mentions_stop_republican(x):
    return DEM if "stop republican" in x.message.lower() else ABSTAIN


lfs = params["lfs"]
lfs = [eval(x) for x in lfs]



applier = PandasLFApplier(lfs=lfs)
L_train = applier.apply(df=df_train)
Y_train = df_train.message_label.values
L_test = applier.apply(df=df_test)
Y_test = df_test.message_label.values



assert np.isnan(L_train).sum(axis=1).max() == 0

#assert np.isnan(Y_train).sum(axis=0).max() == 0

lf_stats = LFAnalysis(L=L_train, lfs=lfs).lf_summary()
print(lf_stats)
# store to lf_stats.json for DVC tracking
#lf_stats.to_json(METRICS_FOLDER / "lf_stats.json")


snorkelled_labels = sum(L_train.max(axis=1) != -1)
print("{}/{} obs ({}%) got some kind of label.".format(
    snorkelled_labels, len(df_train), round(snorkelled_labels/len(df_train), 2)*100))

label_model = LabelModel(cardinality=2, verbose=True)
label_model.fit(L_train=L_train, n_epochs=n_epochs, log_freq=log_freq, seed=seed)
probs_train = label_model.predict_proba(L_train)


# save df_train labelmodel predicts for visualization
df_train["proba"] = probs_train[:, 1]
df_train["predicted_Lmodel"] = probs_to_preds(probs=probs_train)


label_model_acc = label_model.score(L=L_test, Y=Y_test, tie_break_policy="random")[
    "accuracy"
]
print(f"{'Label Model Accuracy:':<25} {label_model_acc * 100:.1f}%")


df_train_filtered, probs_train_filtered = filter_unlabeled_dataframe(
    X=df_train, y=probs_train, L=L_train
)


vectorizer = CountVectorizer(ngram_range=(1, 1))
X_train = vectorizer.fit_transform(df_train_filtered.message.tolist())
X_test = vectorizer.transform(df_test.message.tolist())

preds_train_filtered = probs_to_preds(probs=probs_train_filtered)

sklearn_model = RandomForestClassifier(n_estimators=n_estimators, random_state=seed, verbose=0,
                                       class_weight=class_weight)
print("Predicting labels for {} points, based on {} snorkel-labelled points.".format(X_test.shape[0], X_train.shape[0]))
sklearn_model.fit(X=X_train, y=preds_train_filtered)


prob_predictions = sklearn_model.predict_proba(X_test)[:, 0]
print("Test Accuracy: {} %".format(round(sklearn_model.score(X=X_test, y=Y_test) * 100, 3)))

# save df_train preds for visualization
df_train_filtered["prob"] = sklearn_model.predict_proba(X_train)[:, 0]
df_train_filtered["predicted_label"] = sklearn_model.predict(X_train)
# save df_test preds for visualization
df_test["prob"] = sklearn_model.predict_proba(X_test)[:, 0]
df_test["predicted_label"] = sklearn_model.predict(X_test)


df_train_filtered, probs_train_filtered = filter_unlabeled_dataframe(
    X=df_train, y=probs_train, L=L_train
)



precision, recall, thresholds = precision_recall_curve(Y_test, prob_predictions)
auc = metrics.auc(recall, precision)
print("Test score AUC: {}".format(auc))

df_train.to_parquet(DATA_FOLDER / "df_train_labeled.parquet.gzip", compression="gzip")
df_test.to_parquet(DATA_FOLDER / "df_test_labeled.parquet.gzip", compression="gzip")


# log metrics and params to mlflow
log_metric("auc",auc)
log_param("n_estimators", n_estimators)
log_param("seed", seed)
log_param("n_epochs", n_epochs)
log_param("log_freq", log_freq)
log_param("class_weight", class_weight)
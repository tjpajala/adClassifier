from joblib import dump, load
import pandas as pd
import os
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import yaml
from typing import List, Union, Tuple

from snorkel.labeling import labeling_function, PandasLFApplier, LFAnalysis, filter_unlabeled_dataframe, LabelingFunction
from snorkel.labeling.model import LabelModel
from snorkel.utils import probs_to_preds
from snorkel.preprocess import preprocessor

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve
import sklearn.metrics as metrics

sys.path.append(os.path.relpath(".."))

from mlflow import log_metric, log_param, log_artifacts



params = yaml.safe_load(open('adClassifier/params.yaml'))['labeling_functions']
DATA_FOLDER = Path(params["data_folder"])
df_train_filename = params["df_train_file"]
df_test_filename = params["df_test_file"]
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
    # Return DEM if biden is mentioned
    return DEM if "biden" in x.message.lower() else ABSTAIN

@labeling_function()
def lf_mentions_stop_republican(x):
    return DEM if "stop republican" in x.message.lower() else ABSTAIN

@labeling_function()
def lf_mentions_democrat(x):
    return DEM if "democrat" in x.message.lower() else ABSTAIN

@labeling_function()
def lf_mentions_socialis(x):
    return REP if "socialis" in x.message.lower() else ABSTAIN




lfs = params["lfs"]
lfs = [eval(x) for x in lfs]


def make_L_frames(df_train, df_test, lfs):
    applier = PandasLFApplier(lfs=lfs)
    L_train = applier.apply(df=df_train)
    Y_train = df_train.message_label.values
    L_test = applier.apply(df=df_test)
    Y_test = df_test.message_label.values
    return L_train, Y_train, L_test, Y_test



# store to lf_stats.json for DVC tracking
#lf_stats.to_json(METRICS_FOLDER / "lf_stats.json")


def run_snorkel_labeller(df_train: pd.DataFrame, df_test: pd.DataFrame, lfs: List[LabelingFunction]) -> \
        Tuple[RandomForestClassifier, LabelModel, dict, dict, float, CountVectorizer]:

    L_train, Y_train, L_test, Y_test = make_L_frames(df_train, df_test, lfs)
    assert np.isnan(L_train).sum(axis=1).max() == 0

    lf_stats = LFAnalysis(L=L_train, lfs=lfs).lf_summary()
    print(lf_stats)

    snorkelled_labels = sum(L_train.max(axis=1) != -1)
    print("{}/{} obs ({}%) got some kind of label.".format(
        snorkelled_labels, len(df_train), round(snorkelled_labels / len(df_train), 2) * 100))
    label_model = LabelModel(cardinality=2, verbose=True)
    label_model.fit(L_train=L_train, n_epochs=n_epochs, log_freq=log_freq, seed=seed)
    probs_train = label_model.predict_proba(L_train)
    # save df_train labelmodel predicts for visualization
    preds_train = {"prob":probs_train[:, 0], "predicted_label": probs_to_preds(probs=probs_train)}
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
    print("Predicting labels for {} points, based on {} snorkel-labelled points.".format(X_test.shape[0],
                                                                                         X_train.shape[0]))

    sklearn_model.fit(X=X_train, y=preds_train_filtered)
    prob_predictions = sklearn_model.predict_proba(X_test)[:, 0]
    print("Test Accuracy: {} %".format(round(sklearn_model.score(X=X_test, y=Y_test) * 100, 3)))
    # save df_train preds for visualization


    def get_probs_and_preds(d: np.array, model: RandomForestClassifier) -> dict:
        r = {}
        r["prob"] = model.predict_proba(d)[:,0]
        r["predicted_label"] = model.predict(d)
        return r

    #preds_train = get_probs_and_preds(X_train, sklearn_model)
    #df_train_filtered["prob"] = preds_train["prob"]
    #df_train_filtered["predicted_label"] = preds_train["predicted_label"]
    # save df_test preds for visualization
    preds_test = get_probs_and_preds(X_test, sklearn_model)
    #df_train_filtered, probs_train_filtered = filter_unlabeled_dataframe(
    #    X=df_train, y=probs_train, L=L_train
    #)
    precision, recall, thresholds = precision_recall_curve(Y_test, prob_predictions)
    auc = metrics.auc(recall, precision)
    print("Test score AUC: {}".format(auc))

    # save sklearn model
    dump(sklearn_model, DATA_FOLDER / "models" / "randomforest.model")

    return sklearn_model, label_model, preds_train, preds_test, auc, vectorizer

def predict(X: np.array) -> np.array:
    model_file = Path(DATA_FOLDER / "models" / "randomforest.model")
    if model_file.is_file():
        # file exists
        model = load(model_file)
        return model.predict(X)

    else:
        raise FileNotFoundError("randomforest.model not found, have you trained it first?")

def log_metrics(auc: float) -> None:
    # log metrics and params to mlflow
    log_metric("auc", auc)
    log_param("n_estimators", n_estimators)
    log_param("seed", seed)
    log_param("n_epochs", n_epochs)
    log_param("log_freq", log_freq)
    log_param("class_weight", class_weight)
    return None

def store_results_to_df(df: pd.DataFrame, preds: dict) -> pd.DataFrame:
    df["prob"] = preds["prob"]
    df["predicted_label"] = preds["predicted_label"]
    return df




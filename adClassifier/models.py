import numpy as np
#import keras
import pandas as pd


def image_classifier(df: pd.DataFrame) -> dict:
    r = dict()
    r["prob"] = np.random.random(size=len(df))
    r["predicted_label"] = np.rint(r["prob"])
    return r


def message_classifier(df: pd.DataFrame):

    return


def store_results_to_df(df: pd.DataFrame, preds: dict,
                        prob_colname: str = "prob", pred_colname: str = "predicted_label") -> pd.DataFrame:
    df[prob_colname] = preds["prob"]
    df[pred_colname] = preds["predicted_label"]
    return df

import sys
import os
from joblib import load

sys.path.append(os.path.relpath("."))
from adClassifier.snorkel_labeller import *
from adClassifier.models import image_classifier, store_results_to_df

ABSTAIN = -1
DEM = 0
REP = 1

params = yaml.safe_load(open('adClassifier/params.yaml'))['main_pipeline']
DATA_FOLDER = Path(params["data_folder"])
df_train_filename = params["df_train_file"]
df_test_filename = params["df_test_file"]


df_train = pd.read_parquet(DATA_FOLDER / df_train_filename)
df_test = pd.read_parquet(DATA_FOLDER / df_test_filename)

print('Number of observations in the training data:', len(df_train))
print('Number of observations in the test data:', len(df_test))
print("Classes in train: \n{}".format(df_train.message_label.value_counts()))
print("Classes in test: \n{}".format(df_test.message_label.value_counts()))


preds_train, preds_test, auc = run_snorkel_labeller(df_train=df_train, df_test=df_test, lfs=lfs)
vectorizer = load(DATA_FOLDER / "models" / "vectorizer.model")
X_test = vectorizer.transform(df_test.message.tolist())
px = predict(X_test, Path(DATA_FOLDER / "models" / "randomforest.model"))
print(px)

# save predictions
df_train = store_results_to_df(df_train, preds_train, prob_colname="prob_snorkel", pred_colname="label_snorkel")
df_test = store_results_to_df(df_test, preds_test, prob_colname="prob_snorkel", pred_colname="label_snorkel")
image_train = image_classifier(df_train)
image_test = image_classifier(df_test)
df_train = store_results_to_df(df_train, image_train, prob_colname="prob_image", pred_colname="label_image")
df_test = store_results_to_df(df_test, image_test, prob_colname="prob_image", pred_colname="label_image")
# save prediction results to parquet files
savecols = ["id", "label_snorkel", "prob_snorkel", "prob_image", "label_image"]
df_train[savecols].to_parquet(DATA_FOLDER / "df_train_labeled.parquet.gzip", compression="gzip")
df_test[savecols].to_parquet(DATA_FOLDER / "df_test_labeled.parquet.gzip", compression="gzip")

log_metrics(auc)

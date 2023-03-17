import pandas as pd 
import pickle
import os
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, ConfusionMatrixDisplay
from progress.bar import Bar

# Progress bar
bar = Bar('Creating model           ', max=5)


# Arguments 
parser = argparse.ArgumentParser()

parser.add_argument("--file_name", action="store", dest="file_name", type=str, required=False,
                    help="which dataset?", default = "url_data_vectorized")

parser.add_argument("--data_path", action="store", dest="data_path", type=str, required=False,
                    help="path to vectorized data-set", default = "./data/vectorized")
parser.add_argument("--model_type", action="store", dest="model_type", type=str, required=False,
                    help="model_type", default = "random_forest")

parser.add_argument("--train_split", action="store", dest="train_split", type=float, required=False,
                    help="split of training data", default = 0.3)

parser.add_argument("--rfc_max_dept", action="store", dest="rfc_max_dept", type=int, required=False,
                    help="split of training data", default = None)

parser.add_argument("--rfc_n_estimators", action="store", dest="rfc_n_estimators", type=int, required=False,
                    help="split of training data", default = 100)

args = parser.parse_args()

# Path

path = f"{args.data_path}/{args.file_name}.csv"

data = pd.read_csv(path)

bar.next()


# -------------------------------------
# Model type 1: Random forest classifier
# -------------------------------------

if (args.model_type == "random_forest"):
    
    # Predictor Variables
    x = data[['hostname_length',
       'path_length', 'fd_length', 'tld_length', 'count-', 'count@', 'count?',
       'count%', 'count.', 'count=', 'count-http','count-https', 'count-www', 'count-digits',
       'count-letters', 'count_dir', 'use_of_ip']]
    
    # Target Variable
    y = data['result']

    keys = data[data.label == "malicious"]
    non_keys = data[data.label == "benign"]
    non_keys = non_keys.sample(n=len(keys))

    model_data = pd.concat([keys, non_keys])
    x = model_data[['hostname_length',
       'path_length', 'fd_length', 'tld_length', 'count-', 'count@', 'count?',
       'count%', 'count.', 'count=', 'count-http','count-https', 'count-www', 'count-digits',
       'count-letters', 'count_dir', 'use_of_ip']]
    y = model_data['result']



    # Splitting the data into Training and Testing
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=args.train_split, random_state=42)


    # Classify with random forest 
    rfc = RandomForestClassifier(n_estimators=args.rfc_n_estimators, max_depth=args.rfc_max_dept, random_state=42)
    rfc.fit(x_train, y_train)
    rfc_predictions = rfc.predict(x_test)
    
    # Model performance indicators 
    model_accuracy = accuracy_score(y_test, rfc_predictions)
    model_f1 = f1_score(y_test, rfc_predictions)
    model_confusion_matrix = confusion_matrix(y_test,rfc_predictions)

    bar.next()

# ------------------------------
# Model type 2: Regression model
# ------------------------------

if (args.model_type == "regression"):
    print("Not implemented")





# -------
# Exports 
# -------

# Export data with model scores 
# index, url, label, score (prediction)
x = data[['hostname_length',
       'path_length', 'fd_length', 'tld_length', 'count-', 'count@', 'count?',
       'count%', 'count.', 'count=', 'count-http','count-https', 'count-www', 'count-digits',
       'count-letters', 'count_dir', 'use_of_ip']]

x_predictions = rfc.predict_proba(x)

x_predictions_df = pd.DataFrame(x_predictions, columns=["benign_score", "malicious_score"])


export = data[["url"]].copy()
export["label"] = data["result"]
export["score"] = x_predictions_df["malicious_score"].round(4)
export["label"] = export["label"].replace([0], -1)
export.to_csv('./data/scores/exported_urls.csv', index=False)

bar.next()


#Export pickled model
pickle.dump(rfc, open('./models/model.pickle', 'wb')) # consider joblib
bar.next()

# Export model metadata
# f1, accuracy, confusion_matrix, size 
model_size = joblib_model = os.path.getsize('./models/model.pickle')

model_metadata = {
    "f1": model_f1,
    "accuracy": model_accuracy, 
    "confusion_matrix_0_0": model_confusion_matrix[0,0],
    "confusion_matrix_0_1": model_confusion_matrix[0,1],
    "confusion_matrix_1_0": model_confusion_matrix[1,0],
    "confusion_matrix_1_1": model_confusion_matrix[1,1],
    "size_bits": model_size
}


confusion_plot = ConfusionMatrixDisplay(model_confusion_matrix)

export_meta = pd.DataFrame(data=model_metadata, index=[0])

export_meta.to_csv('./models/model_meta.csv', index=False)
bar.next()

bar.finish()






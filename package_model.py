"""Part D (2/3) - rebuild the final Part C pipeline, save it with joblib and verify that it reloads."""
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from shipment_prep import FEATURES, clean_shipments

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
cfg = pd.read_csv("outputs/part_c_final_config.csv").iloc[0]
max_depth = None if str(cfg["max_depth"]) == "unlimited" else int(cfg["max_depth"])
threshold = float(cfg["threshold"])

table = pd.read_csv("data/modelling_table.csv")
X, y = table[FEATURES], table["is_delayed"]
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp)

model = Pipeline([("selector", SelectKBest(f_classif, k=int(cfg["selector_k"]))),
                  ("model", RandomForestClassifier(random_state=42, n_estimators=int(cfg["n_estimators"]),
                                                   max_depth=max_depth, min_samples_leaf=int(cfg["min_samples_leaf"])))])
model.fit(X_train, y_train)

joblib.dump(model, "models/delay_model.pkl")
pd.DataFrame({"threshold": [threshold]}).to_csv("models/delay_threshold.csv", index=False)
pd.DataFrame({"feature": FEATURES}).to_csv("models/feature_order.csv", index=False)
_, medians = clean_shipments(pd.read_csv("data/raw_shipments.csv"))  # same medians Part A used
pd.DataFrame({"column": list(medians), "median": [round(v, 4) for v in medians.values()]}) \
    .to_csv("models/impute_values.csv", index=False)

reloaded = joblib.load("models/delay_model.pkl")
diff = np.abs(reloaded.predict_proba(X_test)[:, 1] - model.predict_proba(X_test)[:, 1]).max()
pd.DataFrame({"metric": ["test_rows_scored", "reload_max_abs_probability_diff", "threshold"],
              "value": [len(X_test), round(float(diff), 4), threshold]}).to_csv("outputs/part_d_reload_check.csv", index=False)
print("packaged; reload diff =", diff)

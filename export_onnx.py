#!/usr/bin/env python

import json
import os
import pickle

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

os.makedirs("docs", exist_ok=True)

with open("model.bin", "rb") as f:
    pipeline = pickle.load(f)

dv = pipeline.named_steps["vectorizer"]
rf = pipeline.named_steps["classifier"]

# 1. Save the exact ordered feature vocabulary from DictVectorizer
feature_names = dv.get_feature_names_out().tolist()
with open("docs/metadata.json", "w") as f:
    json.dump({"feature_names": feature_names}, f, indent=2)

# 2. Export only the RandomForestClassifier expecting a 1D float vector
initial_type = [("float_input", FloatTensorType([None, len(feature_names)]))]
onnx_model = convert_sklearn(
    rf,
    initial_types=initial_type,
    options={id(rf): {"zipmap": False}},
)

with open("docs/model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

print(f"Exported docs/model.onnx ({len(feature_names)} input features) and docs/metadata.json")

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
import pickle
import category_encoders as ce 

app = Flask(__name__)

model = pickle.load(open("model_lr_cv.pkl", "rb"))
df = pd.read_csv("data_final_clean.csv")
X = df.drop(['is_paid'], axis=1)
encoder = ce.OrdinalEncoder(cols=['region','language','package'])
df_cv = encoder.fit_transform(X)

@app.route("/predict", methods = ["POST"])
def predict():
    json_ = request.json
    cv_json = encoder.transform(json_)
    query = pd.DataFrame(cv_json)
    prediction = model.predict(query)
    predict_proba = model.predict_proba(query)
    proba_list = predict_proba.tolist()
    stack = np.column_stack((proba_list, prediction))
    # return str(ab)
    re_list = stack.tolist()
    return jsonify({"prediction": list(re_list) })
    # return jsonify({"prediction": list(prediction)})
        
if __name__ == "__main__":
    app.run(debug=True)
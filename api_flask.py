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
    cv_query = query.astype({"region":'float', "language":'float', "package":'float', "timezone":'float', "lasttime":'float'})
    # return str(cv_query)
    prediction = model.predict(cv_query)
    return jsonify({"prediction": list(prediction)})
        
if __name__ == "__main__":
    app.run(debug=True)
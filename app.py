import numpy as np
import pandas as pd
from feature_engineering import feature_extraction
from flask import request, Flask, jsonify
from flask_cors import CORS, cross_origin
import xgboost as xgb
import json
import asyncio
from joblib import parallel_backend

app = Flask(__name__)
CORS(app)

# model = joblib.load(r'/Users/Alan/EASP/export_file')
# data_set = np.load('./data/test_set.npy')
# data_dir = "./data/"
# for psv in data_set:
# X_test = xgb.DMatrix(features)

def load_model_predict(x_test, k_fold, path):
    test_pred = np.zeros((x_test.shape[0], k_fold))
    x_test = xgb.DMatrix(x_test)
    for k in range(k_fold):
         model_path_name = path + 'model{}.mdl'.format(k + 1)
         xgb_model = xgb.Booster(model_file=model_path_name)
         try:
            y_test_pred = xgb_model.predict(x_test)
            test_pred[:, k] = y_test_pred
         except:
             test_pred[:, k]= 1
             pass

    test_pred = pd.DataFrame(test_pred)
    result_pro = test_pred.mean(axis=1)

    return result_pro


@app.route('/')
def predict():  # put application's code here
    # query_parameters = request.args
    dvalue = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'SepsisLabel']
    rdata = json.loads(request.args.get('data'))
    # request.json
    for r in rdata:
        for key in dvalue:
            if key not in r:
                r[key] = float("NAN")

    # patient = pd.read_csv(os.path.join(data_dir, "p118826.json"), sep='|')

    # patient = patient.append(df2, ignore_index=True)
    # patient = patient.append(df2, ignore_index=True)
    patient = pd.DataFrame(data=rdata)
    features, labels = feature_extraction(patient)
    # y_test_pred = model.predict(X_test)
    predict_pro =  load_model_predict(features, k_fold=5, path='./model/')
    predictedProbability = np.array(predict_pro)
    # PredictedLabel = [0 if i <= risk_threshold else 1 for i in predict_pro]

    response = jsonify(str(predictedProbability))
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,Origin,X-Requested-With,Accept')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

    # return str(predictedProbability)
# @app.after_request
# def after_request(response):
 # response.headers.add('Access-Control-Allow-Origin', 'localhost:8081')
 # response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,Origin,X-Requested-With,Accept')
 # response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
  # return response


#if __name__ == '__main__':
    # app.run(debug=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

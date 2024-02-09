# import numpy as np
from flask import Flask, request#, jsonify, render_template
from extract_features2 import get_features
import dill
import sys
import os
from joblib import load
import json
import lime
import xgboost
import datetime
from firebase import firebase
# import lime.lime_tabular
# import pandas as pd

# sys.path.append(os.path.join(os.path.dirname(__file__)))
# app = Flask(__name__)

# explainer = dill.load(open('explainer.pkl','rb'))
# model = load("xgb_model_final.dat")


# # @app.route('/')
# # def home():
# #     return render_template('INSERT HOME TEMPLATE')

# @app.route('/predict',methods=['GET'])
# def predict():
#     # wallet_address = str(request.args.get("address"))
#     # features = get_features(wallet_address)
#     features = {'numerical_balance': 0.00120511,'txn_count': 253,'sent_txn': 225,'received_txn': 28,'total_ether_sent': 19.41544801,'max_ether_sent': 1.712009645,'min_ether_sent': 0,'average_ether_sent': 0.085530608,'max_ether_received': 2.7481639,'min_ether_received': 0.032,'avg_ether_received': 0.043794032,'total_ether_received': 11.2112723,'unique_received_from_address': 6,'unique_sent_to_address': 43,'get_time_diff': 621645.1,'mean_time_btw_received': 11609.16352,'mean_time_btw_sent': 2663.328319,'total_erc20_txns': 256,'erc20_total_ether_sent': 18.87144801,'erc20_total_ether_received': 11.15902694,'erc20_uniq_rec_addr': 34,'erc20_uniq_sent_addr': 24}

#     # print(features)
#     prediction = model.predict(features)
#     # print(prediction)
#     # prediction_ouput = str(prediction[0])
#     # output = prediction.map({"1":"Fraud", "0":"Safe"})

#     # exp = explainer.explain_instance(features.squeeze(), model.predict_proba, num_features = 22)
#     # exp = exp.as_html() # bytes -> image

#     return {'prediction': prediction, 'features': json.loads(features.to_json())}
    

# # @app.route('/results',methods=['POST'])
# # def results():

# #     data = request.get_json(force=True)
# #     prediction = model.predict([np.array(list(data.values()))])

# #     output = prediction[0]
# #     return jsonify(output)

# if __name__ == "__main__":
#     app.run(debug=True)

sys.path.append(os.path.join(os.path.dirname(__file__), "Models"))
sys.path.append(os.path.join(os.path.dirname(__file__)))
app = Flask(__name__)

fb_app = firebase.FirebaseApplication('https://crypto-fraud-detection-default-rtdb.asia-southeast1.firebasedatabase.app', None)
explainer = dill.load(open('explainer.pkl','rb'))
model = load("xgb_model_final.dat")


# @app.route('/')
# def home():
#     return render_template('INSERT HOME TEMPLATE')

@app.route('/predict',methods=['GET'])
def predict():
    wallet_address = str(request.args.get("address")).lower()
    name = fb_app.post(f"/user_log", {'start_time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'address': wallet_address, 'type': 'prediction'})
    features = get_features(wallet_address)
    prediction = model.predict(features)
    prediction_ouput = str(prediction[0])
    # output = prediction.map({"1":"Fraud", "0":"Safe"})
    prediction_probabilities = json.dumps(model.predict_proba(features).tolist()[0])
    
    fb_app.patch(f"/user_log/{name['name']}", {'end_time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')})

    # exp = explainer.explain_instance(features.squeeze(), model.predict_proba, num_features = 22)
    # exp = exp.as_html() # bytes -> image

    return {'prediction': prediction_ouput, 'prediction_probabilities': prediction_probabilities, 'features': json.loads(features.to_json())}

@app.route('/api',methods=['GET'])
def results():
    wallet_address = str(request.args.get("address")).lower()    
    name = fb_app.post(f"/user_log", {'start_time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'address': wallet_address, 'type': 'api'})
    features = get_features(wallet_address)
    prediction = model.predict(features)
    prediction_ouput = str(prediction[0])
    prediction_probabilities = json.dumps(model.predict_proba(features).tolist()[0])
    
    fb_app.patch(f"/user_log/{name['name']}", {'end_time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
    
    return {'address': wallet_address, 'prediction': prediction_ouput, 'prediction_probabilities': prediction_probabilities, 'features': json.loads(features.to_json())}


if __name__ == "__main__":
    app.run(debug=True)
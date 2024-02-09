import os
import sys
import pandas as pd
import streamlit as st
import numpy as np
import pandas as pd
import requests
import dill
from joblib import load
import lime
import xgboost
import json
from PIL import Image
import regex as re
# from lime_explainer import explainer
# import argparse
# from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
sys.path.append(os.path.join(os.path.dirname(__file__),'..','Models'))
explainer = dill.load(open('final_explainer.pkl','rb'))
model = load("xgb_model_final.dat")

def get_attributes(text_input):
    # attributes = requests.get(f'https://backendv1-fgxaiirqaq-as.a.run.app/predict?address={text_input.lower()}', timeout=None).json()
    # attributes = requests.get(f'https://backendv1-1-l3f37lfl5a-as.a.run.app/predict?address={text_input.lower().strip()}', timeout=None).json()
    attributes = requests.get(f'https://backendv1-l3f37lfl5a-as.a.run.app/predict?address={text_input.lower().strip()}', timeout=None).json()
    names = ['numerical_balance', 'txn_count', 'sent_txn', 'received_txn', 'total_ether_sent', 'max_ether_sent', 'min_ether_sent', 'average_ether_sent', 'max_ether_received', 'min_ether_received', 'average_ether_received', 'total_ether_received', 'unique_received_from', 'unique_sent_to', 'get_time_diff', 'mean_time_btw_received', 'mean_time_btw_sent', 'total_erc20_txns', 'erc20_total_ether_sent', 'erc20_total_ether_received', 'erc20_unique_rec_add', 'erc20_unique_sent_add']
    dct = {}
    for name in names: 
        dct[name] = attributes['features'][name]['0']
    attributes['features'] = dct

    # with open("lime.png", "rb") as image:
    #     f = image.read()
    #     b = bytearray(f)    
    # row_1_dict = {'numerical_balance': 0.00120511,'txn_count': 253,'sent_txn': 225,'received_txn': 28,'total_ether_sent': 19.41544801,'max_ether_sent': 1.712009645,'min_ether_sent': 0,'average_ether_sent': 0.085530608,'max_ether_received': 2.7481639,'min_ether_received': 0.032,'avg_ether_received': 0.043794032,'total_ether_received': 11.2112723,'unique_received_from_address': 6,'unique_sent_to_address': 43,'get_time_diff': 621645.1,'mean_time_btw_received': 11609.16352,'mean_time_btw_sent': 2663.328319,'total_erc20_txns': 256,'erc20_total_ether_sent': 18.87144801,'erc20_total_ether_received': 11.15902694,'erc20_uniq_rec_addr': 34,'erc20_uniq_sent_addr': 24}
    # attributes = {
    #     'features': row_1_dict,
    #     'prediction': 1,
    #     'lime_bytes': b
    # }
    return attributes

def microservice_description():
    print('To use microservice, have ready a cryptocurrency address and use the following endpoint:\n\
        https://cryptoFraudDetection.com/api/\n\
        ?address={key in address by user}')

def use_regex(input_text):
    pattern = re.compile('|'.join(['numerical_balance', 'txn_count', 'sent_txn', 'received_txn', 'total_ether_sent', 'max_ether_sent', 'min_ether_sent', 'average_ether_sent', 'max_ether_received', 'min_ether_received', 'average_ether_received', 'total_ether_received', 'unique_received_from', 'unique_sent_to', 'get_time_diff', 'mean_time_btw_received', 'mean_time_btw_sent', 'total_erc20_txns', 'erc20_total_ether_sent', 'erc20_total_ether_received', 'erc20_unique_rec_add', 'erc20_unique_sent_add']), re.IGNORECASE)
    return pattern.findall(input_text)[0]

def show_features(attributes, important_feature_names):
    # show feature names and values as a row
    my_dict = attributes['features'].copy()
    if my_dict['txn_count'] == 10000:
        my_dict['txn_count'] = '>10000'
    filtered_dict = {key: my_dict[key] for key in important_feature_names}
    feature_df = pd.DataFrame(pd.Series(filtered_dict,index=filtered_dict.keys())).transpose()
    return feature_df

def show_prediction(attributes):
    if attributes['prediction'] == '1':
        return 'Prediction - Fraud'
    return 'Prediction - Not Fraud'

def show_prediction_probabilities(attributes):
    pred_list = json.loads(attributes['prediction_probabilities'])
    # pred_prob = pd.DataFrame(np.array(pred_list)).transpose()
    # pred_prob.columns = ['Not Fraud Probability','Fraud Probability']
    round_list = [round(x,5) for x in pred_list]
    return round_list

def show_image_and_names(attributes):
    # get image bytes, convert to image, and display
    '''
    image_bytes = attributes['lime_bytes']
    image = Image.open(io.BytesIO(image_bytes))
    '''
    important_features_names = []
    features = attributes['features']
    exp = explainer.explain_instance(np.array(list(features.values())), model.predict_proba, num_features = 3,  labels = [0,1], num_samples = 250)
    image = exp.as_pyplot_figure(label=0)
    
    for important_features in exp.as_list():
        # important_features_names.append(important_features[0].split(' ')[0])
        important_features_names.append(use_regex(important_features[0]))
    
    return important_features_names, image # pyplot

def show_output(address):
    attributes = get_attributes(address)
    important_feature_names, image = show_image_and_names(attributes)
    return {
        'prediction': show_prediction(attributes), 
        'prediction_probabilities': show_prediction_probabilities(attributes),
        'features': show_features(attributes, important_feature_names),
        'image': image
    }

ROOT_DIR = os.path.join(os.path.dirname(__file__), "./")

# Title
st.title("Crypto Fraud Prediction")

tab1, tab2, tab3 = st.tabs(["Prediction", "Microservice", "Contact Us"])
with tab1:

    with st.expander("How to use?"):
        st.markdown("<h4>Key in address into the textbox and press submit. Following items will be displayed:</h4>\
                    1. Is address Fraud/Not Fraud <br>\
                    2. Features related to the address (up to the first 10,000 transactions) <br>\
                    3. The importance of features in making the suggested prediction (Model explanability) <br>\
                    &nbsp; &nbsp; Please wait for a couple of seconds/minutes to see the model prediction!",
                    unsafe_allow_html = True) # can just be replaced in the gr.Column part above
    address = st.text_input(label="Key in address")
    button = st.button("Submit")
    if button:
        # st.stop()

        with st.spinner("Loading your address, please be patient! (May take a few minutes)"):
            output = show_output(address)

            # prediction
            st.markdown('<h4>1. Is fraud address?</h4>', unsafe_allow_html=True)
            # st.text(output['prediction'])
            # st.metric(label = '', value=output['prediction'])
            
            if output['prediction'] == 'Prediction - Fraud':
                prediction_text = f'<p style="font-family:sans-serif; color:red; font-size: 28px;">{output["prediction"]}</p>'
                st.markdown(prediction_text, unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                col1.metric("Fraud Probability✅", output['prediction_probabilities'][1])
                col2.metric("Not Fraud Probability", output['prediction_probabilities'][0])
            else:
                prediction_text = f'<p style="font-family:sans-serif; color:green; font-size: 28px;">{output["prediction"]}</p>'
                st.markdown(prediction_text, unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                col1.metric("Fraud Probability", output['prediction_probabilities'][1])
                col2.metric("Not Fraud Probability✅", output['prediction_probabilities'][0])

            # features
            st.markdown('<hr style="border:2px solid gray">\n<h4>2. Features (Top 3)</h4>', unsafe_allow_html=True)
            st.dataframe(output['features'])

            # model explanability
            st.markdown('<hr style="border:2px solid gray">\n<h4>3. Model Explanability (Top 3)</h4>', unsafe_allow_html=True)
            st.markdown("Please refresh page if image isn't displaying", unsafe_allow_html=True)
            st.pyplot(output['image'])

with tab2:
    # st.markdown('<h4>To use microservice, have ready a cryptocurrency address and use the following endpoint:</h4>\
    #     &nbsp; &nbsp; &nbsp; https://backendv1-l3f37lfl5a-as.a.run.app/api?address={key in address by user}', unsafe_allow_html=True)
    st.markdown('<h4>To use microservice, please submit an enquiry to the team to get the API endpoint</h4>', unsafe_allow_html=True)

with tab3:
    st.markdown('**Contact us via email:** cryptofrauddetection@gmail.com', unsafe_allow_html=True)
    image = Image.open('QR_email.png')
    st.image(image, caption = 'Scan our QR instead')
    st.markdown('**We are more than happy to be of assistance!** <br> Please give 3-5 business days for us to get back to you', unsafe_allow_html=True)






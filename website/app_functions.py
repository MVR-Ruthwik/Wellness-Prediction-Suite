import pickle, os, random
import numpy as np

from sklearn.preprocessing import StandardScaler
import xgboost
import os
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
import numpy as np
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.image import resize



def get_model(path):
    model = load_model(path, compile=False)
    return model

def pred(path):
    # Load and resize the image to match the expected input shape
    img = load_img(path, target_size=(128, 128))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Normalize the pixel values
    img_array = img_array / 255.0
    
    # Predict using the model
    model = get_model('./website/app_models/pneumonia_model.h5')
    predicted = np.round(model.predict(img_array)[0])[0]
    
    return predicted

def ValuePredictor(to_predict_list):
    if len(to_predict_list) == 15:
        page = 'kidney'
        with open('./website/app_models/kidney_model1.pkl', 'rb') as f:
            kidney_model = pickle.load(f)
        pred = kidney_model.predict(np.array(to_predict_list).reshape(-1, len(to_predict_list)))
    elif len(to_predict_list) == 10:
        page = 'liver'
        with open('./website/app_models/liver_model1.pkl', 'rb') as f:
            liver_model = pickle.load(f)
        pred = liver_model.predict(np.array(to_predict_list).reshape(-1, len(to_predict_list)))
    elif len(to_predict_list) == 11:
        page = 'heart'
        with open('./website/app_models/heart_model1.pkl', 'rb') as f:
            heart_model = pickle.load(f)
        pred = heart_model.predict(np.array(to_predict_list).reshape(-1, len(to_predict_list)))
    elif len(to_predict_list) == 9:
        page = 'stroke'
        with open('./website/app_models/avc_scaler.pkl', 'rb') as f:
            stroke_scaler = pickle.load(f)
        l1 = np.array(to_predict_list[2:]).reshape((-1, len(to_predict_list[2:]))).tolist()[0]
        l2 = stroke_scaler.transform(np.array(to_predict_list[0:2]).reshape((-1, 2))).tolist()[0]
        l = l2 + l1
        with open('./website/app_models/avc_model.pkl', 'rb') as f:
            stroke_model = pickle.load(f)
        pred = stroke_model.predict(np.array(l).reshape(-1, len(l)))
    elif len(to_predict_list) == 8:
        page = 'diabete'
        with open('./website/app_models/diabete_model1.pkl', 'rb') as f:
            diabete_model = pickle.load(f)
        pred = diabete_model.predict(np.array(to_predict_list).reshape((-1, 8)))
        print(pred[0], page)
    return pred[0], page

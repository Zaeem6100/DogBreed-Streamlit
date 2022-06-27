import h5py
from keras.models import load_model
from PIL import Image
import numpy as np
from skimage import transform
import streamlit as st


def loadModel(path):
    return load_model(path)

@st.experimental_singleton
def loadLabels(filename):
    with open(filename) as f:
        breed_list = [tuple(map(str, i.split(' '))) for i in f]

    return breed_list


def loadImage(image):
    np_image = Image.open(image)
    np_image = np.array(np_image).astype('float32') / 255
    np_image = transform.resize(np_image, (224, 224, 3))
    #  np_image = np.expand_dims(np_image, axis=0)
    return np_image

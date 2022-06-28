from keras.models import load_model
from PIL import Image
import numpy as np
from skimage import transform
import streamlit as st
import cv2


def loadModel(path):
    return load_model(path)


@st.experimental_singleton
def loadLabels(filename):
    with open(filename) as f:
        breed_list = [tuple(map(str, i.split(' '))) for i in f]
    return breed_list


def loadImage(image):
    # img = cv2.imread(Image.open(image), cv2.IMREAD_COLOR)
    # img = cv2.resize(img, (224, 224))
    # np_image = Image.open(img)
    np_image = np.array(image).astype('float32')
    np_image = transform.resize(np_image, (224, 224, 3))
    np_image = np_image.flatten()
    np_image3 = np.reshape(np_image, (224, 224, 3))
    return np_image3


def img(im):
    image = cv2.imdecode(np.fromstring(im.read(), np.uint8), 1)
    image = cv2.cvtColor(image, cv2.IMREAD_COLOR)
    orig = image.copy()
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (224, 224), (106.0, 177.0, 123.0))
    ig = transform.resize(blob, (1, 224 ,224, 3))
    return (ig)

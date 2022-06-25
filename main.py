import streamlit as st

import helper

path = '/dog_breed_classifier.h5'
filepath = '/file.txt'




if __name__ == '__main__':
    model = helper.loadModel(path)
    breed_list = helper.loadLabels(filename=filepath)

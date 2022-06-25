import streamlit as st

import helper

path = '/dog_breed_classifier.h5'
filepath = '/file.txt'


def predictions(model, breed_list, image):
    probabilities = model.predict(helper.loadImage(image))
    list = []
    for i in probabilities[0].argsort()[-5:][::-1]:
        print(probabilities[0][i], "  :  ", breed_list[i])
        list.append(tuple(probabilities[0][i], breed_list[i]))
    return list


if __name__ == '__main__':
    model = helper.loadModel(path)
    breed_list = helper.loadLabels(filename=filepath)

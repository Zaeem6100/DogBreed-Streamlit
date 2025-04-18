from PIL import Image
import streamlit as st
import helper

path = 'dog_breed_classifier.h5'
filepath = 'file.txt'


def predictions(model, breed_list, image):
    probabilities = model.predict(helper.img(image))
    print(probabilities)
    list = []
    for i in probabilities[0].argsort()[-5:][::-1]:
        print(probabilities[0][i], "  :  ", breed_list[i])
        list.append([probabilities[0][i], breed_list[i]])
    return list


def output():
    st.title("This is for Dog Breed Verify ")
    img_file = st.sidebar.file_uploader(label='Upload a file', type=['png', 'jpg'])
    if img_file:
        if st.button('Check Breed'):
            model = helper.loadModel(path)
            breed_list = helper.loadLabels(filename=filepath)
            list = predictions(model=model, breed_list=breed_list, image=img_file)
            st.write(list)
        else:
            st.write('No Breed ')


if __name__ == '__main__':
    output()

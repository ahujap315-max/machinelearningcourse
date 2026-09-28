import streamlit as st
import pandas as pd
import numpy as np
import sklearn
import pickle

model = pickle.load(open('LinearRegression.pkl', 'rb'))

##lets create web app
st.title("scikit-learn Linear Regression Model")
tv = st.text_input("enter tv sales")
radio = st.text_input("enter radio sales")
newspaper = st.text_input("enter newspaper sales")

if st.button("predict"):
    features = np.array([[tv , radio , newspaper]] , dtype = np.float64)
    results = model.predict(features).reshape(1, -1)
    st.write("predicted sale ::" , results)

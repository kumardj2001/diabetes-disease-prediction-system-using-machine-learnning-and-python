# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 20:36:31 2024

@author: DHEERAJ
"""

import numpy as np
import pickle
import streamlit as st

#loading the saved model
loaded_model = pickle.load(open('C:/Users/DHEERAJ/Desktop/diabities/trained_model.sav','rb'))

# creating a function for prediction

def diabetes_prediction(input_data):

    #changing the input data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance at a time
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)


    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if(prediction[0] == 0):
        return 'The person is not diabitic'
    else:
        return 'The person is diabitic'
    
    
    
def main():
    
    # giving a title 
    st.title('diabetes prediction webapp')
    
    # getting the input data from the user 
    
    Pregnancies = st.text_input('Number of pregnancies')
    Glucose = st.text_input('Glucose level')
    BloodPressure = st.text_input('BloodPressure value')
    SkinThickness = st.text_input('SkinThickness value')
    Insulin = st.text_input('Insulin level')
    BMI = st.text_input('BMI value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    Age = st.text_input('Age of the person')
    
    
    # code for prediction
    diagnosis = ''
    
    # creating a button for prediction 
    
    if st.button('diabetes test result'):
        diagnosis = diabetes_prediction([ Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
        
        
    st.success(diagnosis)
    
    
if __name__ == '__main__':
    main()

    
    
    
    
    
    
    
    
    
    
    
    
    
    



# In[2]:


import streamlit as st


# In[3]:


import pandas as pd


# In[4]:


import numpy as np


# In[5]:


import pickle


# In[6]:


# loading the saved model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))


# In[7]:


input_data = (5,166,72,19,175,25.8,0.587,51)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')


# In[8]:


def diabetes_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)
    if (prediction[0] == 0):
        return('The person is not diabetic')
    else:
        return('The person is diabetic')
    


# In[10]:


def main():
    #giving a title 
    st.title('Diabetes Prediction Web App')
    st.text('Made By Raye Haarika')

    
    #input data fields
    Pregnancies=st.text_input("number of pregnancies")
    Glucose=st.text_input("Glucose level")
    BloodPressure=st.text_input("blood pressure value:")
    SkinThickness=st.text_input("Skin thickness value")
    Insulin=st.text_input("Insulin Level")
    BMI=st.text_input("BMI VALUE")
    DiabetesPedigreeFunction=st.text_input("diabetes pedigree function")
    Age=st.text_input("age")
    
    #code for predicition
    diagnosis=''
    
    #creating a button for prediction
    if st.button("Diabetes test result"):
        diagnosis=diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
    
    st.success(diagnosis)

if __name__ == '__main__':
    main()


import numpy as np
import pandas as pd
import streamlit as st
import joblib
from tensorflow.keras.models import load_model # type: ignore

# Load the model and preprocessing tools
model = load_model('iris_model.keras')
scaler = joblib.load('scaler.joblib')
encoder = joblib.load('encoder.joblib')

st.title('Iris Species Prediction App')
st.write('Enter the measurements of the Iris flower to predict its species.')

# Input features from the user
sepal_length = st.slider('Sepal Length (cm)', 4.0, 8.0, 5.0)
sepal_width = st.slider('Sepal Width (cm)', 2.0, 4.5, 3.0)
petal_length = st.slider('Petal Length (cm)', 1.0, 7.0, 4.0)
petal_width = st.slider('Petal Width (cm)', 0.1, 2.5, 1.0)

# Create a predict button
predict_button = st.button('Predict')

if predict_button:
    # Create a DataFrame from user inputs
    input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                                columns=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'])

    # Scale the input data
    scaled_input = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(scaled_input)
    predicted_class_index = np.argmax(prediction, axis=1)[0]

    # Decode the prediction
    predicted_species = encoder.inverse_transform([predicted_class_index])[0]

    st.success(f'The predicted Iris species is: {predicted_species}')

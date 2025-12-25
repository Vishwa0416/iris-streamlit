import streamlit as st
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load model
with open("iris_model.pkl", "rb") as f:
    model = pickle.load(f)

# Title
st.title("Iris Species Predictor 🌸")

# Input features
st.sidebar.header("Enter Flower Features")
sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.0)
sepal_width = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 4.0)
petal_width = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 1.0)

# Predict button
if st.button("Predict"):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)[0]
    species = ["Setosa", "Versicolor", "Virginica"]
    st.success(f"The predicted species is: **{species[prediction]}**")

# Optional: display dataset
if st.checkbox("Show Iris Dataset"):
    from sklearn.datasets import load_iris
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = [iris.target_names[i] for i in iris.target]
    st.dataframe(df)
    
    # Simple pairplot
    st.write("### Pairplot of Iris Dataset")
    sns.pairplot(df, hue="species")
    st.pyplot(plt)

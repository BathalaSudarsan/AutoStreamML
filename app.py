import os
import pandas as pd
import streamlit as st
from ydata_profiling import ProfileReport
from pycaret.regression import setup, compare_models, pull, save_model

# Check if dataset exists
if os.path.exists('dataset.csv'):
    df = pd.read_csv('dataset.csv', index_col=None)

with st.sidebar:
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title("AutoStreamML")
    choice = st.radio("Navigation", ["Upload", "Profiling", "Modelling", "Download"])
    st.info("This project application helps you build and explore your data.")

if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)

if choice == "Profiling":
    st.title("Exploratory Data Analysis")
    if 'df' in locals():
        profile = ProfileReport(df, title='Exploratory Data Analysis Report')
        st.write(profile.to_html())
    else:
        st.warning("Please upload a dataset first.")

if choice == "Modelling":
    st.title("Modeling")
    if 'df' in locals():
        chosen_target = st.selectbox('Choose the Target Column', df.columns)
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if st.button('Run Modelling'):
            try:
                # Initialize setup
                exp_reg = setup(data=df, target=chosen_target, categorical_features=categorical_cols, verbose=False)
                setup_df = pull()
                st.dataframe(setup_df)

                # Compare models and select the best one
                best_model = compare_models()
                compare_df = pull()
                st.dataframe(compare_df)

                # Display the best model
                st.write(f"The best model is: {best_model}")

                # Save the best model
                save_model(best_model, 'best_model')
                st.success("Modeling completed and best model saved as 'best_model.pkl'.")
            except Exception as e:
                st.error(f"An error occurred during the modeling process: {e}")
    else:
        st.warning("Please upload a dataset first.")

if choice == "Download":
    with open('best_model.pkl', 'rb') as f:
        st.download_button('Download Model', f, file_name="best_model.pkl")

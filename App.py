import streamlit as st
import numpy as np
from sklearn.preprocessing import FunctionTransformer
from main import OutlierRemoval
import pickle


st.set_page_config(page_title='Music Popularity Predictor',page_icon="🎵")
st.title("🎶 Music Popularity Predictor")
st.markdown("Predict how popular a song could be based on its audio fetures.")

st.sidebar.title('Navigation')
page=st.sidebar.radio("Go to",['Predict Popularity','About'])

#['Energy', 'Valence', 'Danceability', 'Loudness', 'Acousticness','Duration (ms)']

if page=="Predict Popularity":
    st.header("🔍 Enter Song Features")
    col1,col2=st.columns(2)

    with col1:
        energy = st.slider("Energy", 0.0, 1.0, 0.6)
        danceability = st.slider("Danceability", 0.0, 1.0, 0.5)
        loudness = st.slider("Loudness (dB)", -60.0, 0.0, -10.0)
        acousticness = st.slider("Acousticness", 0.0, 1.0, 0.3)
        tempo = st.slider("Tempo (BPM)", 0.0, 250.0, 120.0)

    with col2:
        valence = st.slider("Valence", 0.0, 1.0, 0.5)
        speechiness = st.slider("Speechiness", 0.0, 1.0, 0.1)
        liveness = st.slider("Liveness", 0.0, 1.0, 0.2)
        duration_ms = st.number_input("Duration (ms)", min_value=0, max_value=600000, value=200000, step=1000)
    
    if st.button("🎧 Predict Popularity"):
        features=np.array([energy,valence,danceability,loudness,acousticness,tempo,speechiness,liveness,duration_ms]).reshape(1,9)
        fc = FunctionTransformer(OutlierRemoval)
        model= pickle.load(open('Model.pkl', 'rb'))
        prediction=model.predict(features)
        st.success(f"🎉 Predicted Popularity Score: {int(prediction[0])} / 100")

elif page == "About":
    st.header("ℹ️ About")
    st.markdown("""
    Welcome to the **Music Popularity Predictor**! 🎵

    This web application allows users to **predict the popularity of a song** based on its audio features. 
    It uses a Machine Learning model trained on real-world music data to estimate a score that represents how likely a song is to become popular.

    ### 🔍 How It Works
    - You input various audio features like **danceability**, **energy**, **tempo**, and **valence**, **loudness**, **Duration**.
    - The model analyzes these features and predicts a **popularity score** out of 100.
    - This prediction can help artists, producers, and music lovers gauge a song's potential appeal.

    ### 🧠 Technologies Used
    - **Python**
    - **Streamlit** for the web interface
    - **Pandas & NumPy** for data handling
    - **Matplotlib & Seaborn** for data visualization
    - **Scikit-learn** for Machine Learning
    - **Joblib** to load trained models

    

    ---
    Made with ❤️ using Streamlit
    """)

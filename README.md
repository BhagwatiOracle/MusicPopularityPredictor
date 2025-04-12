# 🎶 Music Popularity Predictor
The Music Popularity Predictor is a web app built with Streamlit that uses a trained machine learning model to estimate a song’s popularity (on a scale of 0 to 100) based on its audio characteristics.

![Image Link](![Screenshot 2025-04-12 234037.png](Screenshot%202025-04-12%20234037.png))
_________

## 🧠 Features Used for Prediction
🎧 Energy

😊 Valence

💃 Danceability

🔊 Loudness (dB)

🎻 Acousticness

🎵 Tempo (BPM)

🗣️ Speechiness

🧍‍♂️ Liveness

⏱️ Duration (ms)

_______

# Development Workflow

## 1.EDA
- Analyzed distribution of categorical & numerical features
- Visualized correlation between different features.

## 2.Feature Engineering
- Feature Scaling
- Standardization
- Outlier Removal

## 4. Model Training
Different Machine Learning algorithims are trained on data -
1.Linear Regression
2.Ridge Regression
3.Lasso Regression
4.DecisionTree Regressor
5.Voting Regressor
6.RandomForest Regressor
7.XGBoost Regressor

## 4. Hyperparameter Tuning

- Used GridSearchCV and RandomizedSearchCV to tune differnt parameters.


![Image Link](![Screenshot 2025-04-12 234127.png](Screenshot%202025-04-12%20234127.png))

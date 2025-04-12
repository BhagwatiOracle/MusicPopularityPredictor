from sklearn.preprocessing import FunctionTransformer
import pandas as pd
import numpy as np
def OutlierRemoval(X):
  features=['Energy','Valence','Danceability','Loudness','Acousticness','Tempo','Speechiness','Liveness','Duration (ms)']

  df=pd.DataFrame(X,columns=features)
  Q1=np.percentile(X,25)
  Q3=np.percentile(X,75)
  iqr=Q3-Q1
  upper_limit = Q3 + 1.5 * iqr
  lower_limit = Q1 - 1.5 * iqr

  df=np.where(df>upper_limit,upper_limit,np.where(X<lower_limit,lower_limit,X))
  return df

fc=FunctionTransformer(OutlierRemoval)
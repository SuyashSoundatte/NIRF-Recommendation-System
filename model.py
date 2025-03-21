import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

df = pd.read_csv(r"E:\Recommendation_system\Dataset\NIRF_Dataset_Random.csv")


df['Ranking'] = np.rint(df['Ranking']).astype(int)

ranking_col = "Ranking" 
print(df["Ranking"])
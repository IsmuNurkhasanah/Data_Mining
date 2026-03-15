import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

#load data set
df = pd.read_csv("Titanic-Dataset.csv")
df.head()

#identifikasi data
df.info()
df.isnull().sum()

#data cleaning
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

#hapus colom
df.drop(['Cabin','Ticket','Name'], axis=1, inplace=True)

#encoding data
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
df['Embarked'] = le.fit_transform(df['Embarked'])

#normalisasi data
scaler = MinMaxScaler()
df[['Age','Fare']] = scaler.fit_transform(df[['Age','Fare']])

#tampilan akhir
df.info()
df.head()
df.describe()

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import pickle

df = pd.read_csv('Accidentes\indices-de-estadisticas-de-accidentes-viales-apodaca(2)(1).csv')

#Eliminacion de valor nulo en columna Hora
df = df.drop(df[df[' Hora'].isnull()].index).reset_index()

indexes = df[df[' Hora']=='S/D'].index
df = df.drop(indexes).reset_index()

indexes = []
for row in range(df.shape[0]):
  if(' ' in str(df.loc[row,' Hora'])):
    indexes.append(row)
df = df.drop(indexes)

# Juntar las columnas 'Fecha' y 'Hora' en una sola columna 'Fecha_Hora'
df['Fecha_Hora'] = pd.to_datetime(df[' Fecha'] + ' ' + df[' Hora'], format='%Y-%m-%d %H:%M:%S')

df_resample = df.set_index('Fecha_Hora')
df_resample = pd.DataFrame(df_resample.resample('4H').count().iloc[:,1])

df_resample.columns = ['Cantidad']

#Generacion de nuevos parametros a partir de datos anteriores y siguientes
feature='Cantidad'
df_resample['cant_p1'] = df_resample[feature].shift(1)
df_resample['cant_p2'] = df_resample[feature].shift(2)
df_resample['cant_p3'] = df_resample[feature].shift(3)
df_resample['cant_p4'] = df_resample[feature].shift(4)
df_resample = df_resample.fillna(0)

df_resample['rolling_window_mean'] = df_resample['cant_p1'].rolling(window=15).mean()
df_resample['rolling_window_mean7'] = df_resample['cant_p1'].rolling(window=20).mean()
df_resample['rolling_window_max'] = df_resample['cant_p1'].rolling(window=10).max()
df_resample['rolling_window_min'] = df_resample['cant_p1'].rolling(window=15).min()
df_resample.dropna(inplace=True)

test_size = int(df_resample.shape[0]*0.1)
X_train = df_resample.drop(columns=[feature])[:-test_size]
y_train = df_resample[[feature]][:-test_size]
X_test= df_resample.drop(columns=[feature])[-test_size:]
y_test = df_resample[[feature]][-test_size:]


regr = RandomForestRegressor(n_estimators=1000, max_depth=150, random_state=42)
regr.fit(X_train, y_train)
predict_test = pd.Series(regr.predict(X_test), y_test.index, name='diff_prediction')
mse = mean_squared_error(predict_test, y_test, squared=False)
print(f'The model MSE is {mse}')

r2_score_val = r2_score(predict_test, y_test)
print(f'R2 Score: {r2_score_val}')

pickle.dump(regr, open('Accidentes/model.pkl','wb'))
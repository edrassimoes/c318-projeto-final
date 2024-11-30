# Bibliotecas Gerais
import pandas as pd
import numpy as np
import math

# Sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Importando o dataset
df = pd.read_csv('./data/gym_members_exercise_tracking.csv')

# ---- Limpeza dos Dados (Data Cleaning)

# Remoção de dados com valores vazios
n_dados = df.shape[0]

df.dropna(inplace=True)

print(f"Antes da limpeza {n_dados}")
print(f"Depois da limpeza {df.shape[0]}")
print(f"Número de dados vazios: {n_dados - df.shape[0]}")

# ---- Normalização e Scaling dos dados

# Normalização dos valores
scaler = StandardScaler()
colunas_para_scaling = [
    "Age",
    "Weight (kg)",
    "Height (m)",
    "Avg_BPM",
    "Session_Duration (hours)",
    "Fat_Percentage",
    "Water_Intake (liters)",
    "Workout_Frequency (days/week)",
    "Max_BPM",
    "Resting_BPM",
    "BMI",
    "Experience_Level",
]

scaled_data = scaler.fit_transform(df[colunas_para_scaling])

df[colunas_para_scaling] = scaled_data

# Transformação das colunas de texto para variáveis "dummy"
colunas_texto = ["Gender", "Workout_Type"]
df_tmp = pd.get_dummies(df[colunas_texto], columns=colunas_texto)

df.drop(columns=colunas_texto, inplace=True)
df = pd.concat([df, df_tmp], axis=1)

# ---- Preparação dos Dados para Modelagem (Machine Learning)

'''
  Alguns dados são removidos da análise por não ser possível obtê-los apenas com uma análise da pessoa

  Por exemplo:
    Batimentos cardíacos e calorias requerem aparelhos para serem medidos
'''

# Dados que serão usados para cada um dos modelos
df_calories = df.drop(columns=["Max_BPM", "Resting_BPM","Avg_BPM"])

# Separação dos dados para treinamento e testes
df_train_calories, df_test_calories = train_test_split(df_calories, test_size=0.2, random_state=42)

# Separação entre variáveis de entrada e saída
df_calories = {
    "x_train": df_train_calories.drop(columns=["Calories_Burned"]),
    "y_train": df_train_calories["Calories_Burned"],
    "x_test": df_test_calories.drop(columns=["Calories_Burned"]),
    "y_test": df_test_calories["Calories_Burned"]
}

# Criação do modelo de regressão linear
model = linear_model.LinearRegression()

# Treinamento do modelo

model.fit(df_calories["x_train"], df_calories["y_train"])

# Predição da saída

y_predict_calories = model.predict(df_calories["x_test"])

# Precisão do treinamento
erro_q_medio = mean_squared_error(df_calories["y_test"], y_predict_calories)
coef_det = r2_score(df_calories["y_test"], y_predict_calories)

print("Modelo sem dados de Batimento Cardíaco")
print(f"Erro quadrático médio: {round(erro_q_medio, 2)}")
print(f"Erro médio: +-{round(math.sqrt(erro_q_medio), 2)} Kcal")
print(f"Coeficiente de determinação: {round(coef_det, 2)}")

# Modelagem dos dados sem exclusão de campos
'''
  Precisão de um cenário onde é possível utilizar os batimentos da pessoa para calcular o consumo de calorias
'''
df_calories_sim = df.copy()
df_train_calories_sim, df_test_calories_sim = train_test_split(df_calories_sim, test_size=0.2, random_state=42)
df_calories_sim = {
    "x_train": df_train_calories_sim.drop(columns=["Calories_Burned"]),
    "y_train": df_train_calories_sim["Calories_Burned"],
    "x_test": df_test_calories_sim.drop(columns=["Calories_Burned"]),
    "y_test": df_test_calories_sim["Calories_Burned"]
}

model = linear_model.LinearRegression()

# Treinamento do modelo

model.fit(df_calories_sim["x_train"], df_calories_sim["y_train"])

# Predição da saída

y_predict_calories = model.predict(df_calories_sim["x_test"])

# Precisão do treinamento
erro_q_medio_sim = mean_squared_error(df_calories_sim["y_test"], y_predict_calories)
coef_det_sim = r2_score(df_calories_sim["y_test"], y_predict_calories)

print("\nModelo com todos os dados disponíveis")
print(f"Erro quadrático médio: {round(erro_q_medio_sim, 2)}")
print(f"Erro médio: +-{round(math.sqrt(erro_q_medio_sim), 2)} Kcal")
print(f"Coeficiente de determinação: {round(coef_det_sim, 2)}")

print("\nComparação:")
print(f"Erro quadrático médio: {round(erro_q_medio, 2)} x {round(erro_q_medio_sim, 2)} ({round((erro_q_medio_sim/erro_q_medio)*100, 2)}%)")
print(f"Erro médio: {round(math.sqrt(erro_q_medio), 2)} x {round(math.sqrt(erro_q_medio_sim), 2)} ({round((math.sqrt(erro_q_medio_sim)/math.sqrt(erro_q_medio))*100, 2)}%)")
print(f"Coeficiente de determinação: {round(coef_det, 2)} x {round(coef_det_sim, 2)}")
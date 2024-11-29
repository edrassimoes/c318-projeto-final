from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------- Inicialização e Preprocessamento -----------------------------

# Carrega o arquivo CSV em um DataFrame
file_path = 'data/gym_members_exercise_tracking.csv'
data = pd.read_csv(file_path)

# Colunas contínuas
continuous_columns = ['Age', 'Weight (kg)', 'Height (m)', 'Max_BPM', 'Avg_BPM',
                      'Resting_BPM', 'Session_Duration (hours)', 'Calories_Burned',
                      'Fat_Percentage', 'Water_Intake (liters)', 'BMI', 'Workout_Frequency (days/week)']

# Calcula a matriz de correlação
corr_matrix = data[continuous_columns].corr()

# Plota o heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
plt.title('Matriz de Correlação das Variáveis Contínuas')
plt.show()

# Encoding de variáveis categóricas
data['Gender'] = LabelEncoder().fit_transform(data['Gender'])
data['Experience_Level'] = LabelEncoder().fit_transform(data['Experience_Level'])
data = pd.get_dummies(data, columns=['Workout_Type'], drop_first=False)

# Atualiza as colunas categóricas
categorical_columns = [col for col in data.columns if 'Workout_Type' in col] + ['Gender', 'Experience_Level']

# Escalonamento das variáveis contínuas (ANTES de detectar outliers)
scaler_before_outlier = StandardScaler()
scaled_continuous_data = scaler_before_outlier.fit_transform(data[continuous_columns])

# Aplica o modelo de Isolation Forest para detectar outliers
isolation_forest = IsolationForest(random_state=42, contamination=0.05)
outlier_pred = isolation_forest.fit_predict(scaled_continuous_data)

# Remove os outliers
data['Outlier'] = outlier_pred
cleaned_data = data[data['Outlier'] == 1].drop(columns=['Outlier'])
cleaned_data = cleaned_data.reset_index(drop=True)

# ----------------------------- Cluster de Peso e Gordura -------------------------------------

# Seleção das variáveis Peso e Gordura
peso_gordura_columns = ['Weight (kg)', 'Fat_Percentage']

# Dados SEM OUTLIERS para Peso e Gordura
cleaned_peso_gordura_data = cleaned_data[peso_gordura_columns]

# Escalonamento dos dados limpos
scaler_after_outlier = StandardScaler()
scaled_cleaned_data = scaler_after_outlier.fit_transform(cleaned_peso_gordura_data)

# Aplica KMeans para segmentar os dados sem outliers
kmeans_cleaned_peso_gordura = KMeans(n_clusters=3, random_state=42)
cleaned_data['Peso_Gordura_Cluster'] = kmeans_cleaned_peso_gordura.fit_predict(scaled_cleaned_data)

# Inverte o escalonamento para os valores originais
original_cleaned_data = scaler_after_outlier.inverse_transform(scaled_cleaned_data)

# Adiciona as colunas de valores originais ao DataFrame limpo
cleaned_data['Peso_Original'] = original_cleaned_data[:, 0]
cleaned_data['Gordura_Original'] = original_cleaned_data[:, 1]

# ----------------------------- Visualização do Cluster -------------------------------------

# Exibe o gráfico com os valores originais sem outliers
plt.figure(figsize=(10, 6))
plt.scatter(cleaned_data['Peso_Original'], cleaned_data['Gordura_Original'], c=cleaned_data['Peso_Gordura_Cluster'], cmap='viridis')
plt.xlabel('Peso (kg)')
plt.ylabel('Percentual de Gordura')
plt.title('Clusters de Peso e Gordura (Sem Outliers)')
plt.colorbar(label='Cluster')
plt.show()

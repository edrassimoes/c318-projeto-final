import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

"""
Predição de Cancelamento de Matrícula (Churn Prediction)
"""

# Carrega o dataset
file_path = 'data/cleaned_dataset.csv'
data = pd.read_csv(file_path)

# Define a variável-alvo (exemplo: churn = frequência < 2 dias/semana)
data['Churn'] = (data['Workout_Frequency (days/week)'] < 2).astype(int)

# Remove colunas desnecessárias, se existirem
columns_to_drop = ['Churn', 'Cluster']
existing_columns = [col for col in columns_to_drop if col in data.columns]
X = data.drop(columns=existing_columns)

# Alvo
y = data['Churn']

# Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Modelo de classificação
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

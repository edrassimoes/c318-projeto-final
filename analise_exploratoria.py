import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carrega o dataset
file_path = 'data/gym_members_exercise_tracking.csv'  # Altere se necessário
data = pd.read_csv(file_path)

# Gráfico 1: Proporção por Gênero (pizza)
gender_counts = data['Gender'].value_counts()
gender_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=['#ff9999', '#66b3ff'])
plt.title('Distribuição por Gênero')
plt.ylabel('')  # Remove o rótulo do eixo Y
plt.show()

# Gráfico 2: Frequência de Treinos por Semana (pizza)
frequency_counts = data['Workout_Frequency (days/week)'].value_counts()
frequency_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'))
plt.title('Frequência de Treinos por Semana')
plt.ylabel('')
plt.show()

# Gráfico 3: Tipos de Treinos (pizza)
workout_type_counts = data['Workout_Type'].value_counts()
workout_type_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'))
plt.title('Tipos de Treinos')
plt.ylabel('')
plt.show()

# Gráfico 4: Histogramas (Idade, Fat_Percentage, Session_Duration, Calories_Burned)
columns_to_plot = ['Age', 'Fat_Percentage', 'Session_Duration (hours)', 'Calories_Burned']
colors = ['blue', 'green', 'orange', 'purple']

for column, color in zip(columns_to_plot, colors):
    sns.histplot(data[column], kde=True, color=color, bins=20)
    plt.title(f'Histograma de {column.replace("_", " ")}')
    plt.xlabel(column.replace('_', ' '))
    plt.ylabel('Frequência')
    plt.tight_layout()
    plt.show()

# Colunas contínuas
continuous_columns = ['Age', 'Weight (kg)', 'Height (m)', 'Max_BPM', 'Avg_BPM',
                      'Resting_BPM', 'Session_Duration (hours)', 'Calories_Burned',
                      'Fat_Percentage', 'Water_Intake (liters)', 'BMI', 'Workout_Frequency (days/week)']

# Calcula a matriz de correlação
corr_matrix = data[continuous_columns].corr()

# Plota o heatmap
plt.figure(figsize=(12, 10))  # Aumenta o tamanho da figura
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1, cbar_kws={'shrink': 0.8})
plt.title('Matriz de Correlação das Variáveis Contínuas', fontsize=16)
plt.tight_layout()
plt.xticks(rotation=45, ha='right')
plt.show()

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import IsolationForest

"""
Informações sobre o dataset  
    - 1. Age: Age of the gym member.
    - 2. Gender: Gender of the gym member (Male or Female).
    - 3. Weight (kg): Member’s weight in kilograms.
    - 4. Height (m): Member’s height in meters.
    - 5. Max_BPM: Maximum heart rate (beats per minute) during workout sessions.
    - 6. Avg_BPM: Average heart rate during workout sessions.
    - 7. Resting_BPM: Heart rate at rest before workout.
    - 8. Session_Duration (hours): Duration of each workout session in hours.
    - 9. Calories_Burned: Total calories burned during each session.
    - 10. Workout_Type: Type of workout performed (e.g., Cardio, Strength, Yoga, HIIT).
    - 11. Fat_Percentage: Body fat percentage of the member.
    - 12. Water_Intake (liters): Daily water intake during workouts.
    - 13. Workout_Frequency (days/week): Number of workout sessions per week.
    - 14. Experience_Level: Level of experience, from beginner (1) to expert (3).
    - 15. BMI: Body Mass Index, calculated from height and weight.
"""

# Carrega o arquivo CSV em um DataFrame
file_path = 'data/gym_members_exercise_tracking.csv'
data = pd.read_csv(file_path)

# Colunas contínuas
continuous_columns = ['Age', 'Weight (kg)', 'Height (m)', 'Max_BPM', 'Avg_BPM',
                      'Resting_BPM', 'Session_Duration (hours)', 'Calories_Burned',
                      'Fat_Percentage', 'Water_Intake (liters)', 'BMI']

# Encoding de 'Gender' e 'Workout_Type'
data['Gender'] = LabelEncoder().fit_transform(data['Gender'])
data = pd.get_dummies(data, columns=['Workout_Type'])

# Atualiza as colunas categóricas
categorical_columns = [col for col in data.columns if 'Workout_Type' in col] + ['Gender']

# Escalonamento das variáveis contínuas
scaler = StandardScaler()
scaled_continuous_data = scaler.fit_transform(data[continuous_columns])

# Aplica o modelo de Isolation Forest
isolation_forest = IsolationForest(random_state=42, contamination=0.05)
outlier_pred = isolation_forest.fit_predict(scaled_continuous_data)

# Remove os outliers
data['Outlier'] = outlier_pred
cleaned_data = data[data['Outlier'] == 1].drop(columns=['Outlier'])

# Reseta os índices após a remoção dos outliers
cleaned_data = cleaned_data.reset_index(drop=True)

# Salva o DataSet final
output_file_path = 'data/cleaned_dataset.csv'
cleaned_data.to_csv(output_file_path, index=False)

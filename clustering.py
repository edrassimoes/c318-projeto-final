import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Carrega o dataset
file_path = 'data/cleaned_dataset.csv'
data = pd.read_csv(file_path)

# Seleciona variáveis para clusterização
cluster_data = data[['Age', 'Workout_Frequency (days/week)', 'Calories_Burned']]

# Modelo de K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(cluster_data)

# Adiciona os clusters ao DataFrame
data['Cluster'] = clusters

# Visualiza os clusters
plt.scatter(data['Age'], data['Workout_Frequency (days/week)'], c=clusters, cmap='viridis', alpha=0.7)
plt.xlabel('Age')
plt.ylabel('Workout Frequency (days/week)')
plt.title('Clusters de Perfis de Clientes')
plt.colorbar(label='Cluster')
plt.show()

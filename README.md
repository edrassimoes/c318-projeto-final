# Clustering de Peso e Gordura Corporal (Sem Outliers)

Este projeto aplica técnicas de **Machine Learning** para realizar **detecção de outliers** e **clustering** de dados relacionados a peso e percentual de gordura corporal. O objetivo é identificar padrões nos dados, agrupando-os em clusters e removendo anomalias para obter resultados mais precisos.

## 📋 Funcionalidades

1. **Encoding de variáveis categóricas** para facilitar o uso de algoritmos de machine learning.
2. **Detecção e remoção de outliers** utilizando o modelo **Isolation Forest**.
3. **Normalização e escalonamento dos dados** com o **StandardScaler**.
4. **Clustering** dos dados de peso e gordura corporal com o algoritmo **KMeans**.
5. **Visualização** dos clusters em um gráfico.

## 🚀 Tecnologias Utilizadas

- **Python**
- **Bibliotecas**:
  - `pandas` para manipulação de dados.
  - `numpy` para operações numéricas.
  - `matplotlib` para visualização.
  - `scikit-learn` para modelagem (Isolation Forest, KMeans e escalonamento).

## 📂 Estrutura do Código e Trechos Relevantes

### 1. **Pré-processamento: Encoding de Variáveis**

- **Encoding de dados categóricos**:
  Antes de aplicar o pipeline, as variáveis categóricas são transformadas em colunas binárias ou numéricas para que os algoritmos possam processá-las. Por exemplo, a coluna `Workout_Type` foi subdividida em múltiplas colunas, uma para cada tipo de treino.

  ```python
  data['Gender'] = LabelEncoder().fit_transform(data['Gender'])
  data['Experience_Level'] = LabelEncoder().fit_transform(data['Experience_Level'])
  data = pd.get_dummies(data, columns=['Workout_Type'], drop_first=False)
  ```

Esse processo facilita o treinamento de modelos como o **Isolation Forest** e o **KMeans**.

### 2. **Detecção e Remoção de Outliers**

- **Escalonamento dos dados contínuos** antes da detecção de outliers para garantir que todas as variáveis tenham a mesma escala:

  ```python
  scaler_before_outlier = StandardScaler()
  scaled_continuous_data = scaler_before_outlier.fit_transform(data[continuous_columns])
  ```

- **Aplicação do modelo Isolation Forest** para identificar e remover outliers:

  ```python
  isolation_forest = IsolationForest(random_state=42, contamination=0.05)
  outlier_pred = isolation_forest.fit_predict(scaled_continuous_data)

  data['Outlier'] = outlier_pred
  cleaned_data = data[data['Outlier'] == 1].drop(columns=['Outlier'])
  cleaned_data = cleaned_data.reset_index(drop=True)
  ```

### 3. **Clustering de Peso e Gordura**

- **Seleção e escalonamento das variáveis de peso e gordura** após a remoção de outliers:

  ```python
  peso_gordura_columns = ['Weight (kg)', 'Fat_Percentage']
  cleaned_peso_gordura_data = cleaned_data[peso_gordura_columns]

  scaler_after_outlier = StandardScaler()
  scaled_cleaned_peso_gordura_data = scaler_after_outlier.fit_transform(cleaned_peso_gordura_data)
  ```

- **Aplicação do KMeans** para agrupar os dados em clusters:

  ```python
  kmeans_cleaned_peso_gordura = KMeans(n_clusters=3, random_state=42)
  cleaned_data['Peso_Gordura_Cluster'] = kmeans_cleaned_peso_gordura.fit_predict(scaled_cleaned_peso_gordura_data)
  ```

- **Reconversão dos dados escalonados para os valores originais**:

  ```python
  original_cleaned_peso_gordura_data = scaler_after_outlier.inverse_transform(scaled_cleaned_peso_gordura_data)
  cleaned_data['Peso_Original'] = original_cleaned_peso_gordura_data[:, 0]
  cleaned_data['Gordura_Original'] = original_cleaned_peso_gordura_data[:, 1]
  ```

### 4. **Visualização dos Clusters**

O gráfico gerado exibe os clusters identificados com base nos dados sem outliers:

```python
plt.figure(figsize=(10, 6))
plt.scatter(cleaned_data['Peso_Original'], cleaned_data['Gordura_Original'], c=cleaned_data['Peso_Gordura_Cluster'], cmap='viridis')
plt.xlabel('Peso (kg)')
plt.ylabel('Percentual de Gordura')
plt.title('Clusters de Peso e Gordura (Sem Outliers)')
plt.colorbar(label='Cluster')
plt.show()
```

## 📊 Visualização
- **Eixos**:
  - Eixo X: Peso (kg)
  - Eixo Y: Percentual de Gordura
- **Cores**: Representam os diferentes clusters criados pelo algoritmo KMeans.

![Gráfico de Clusters](figures/clusters_peso_gordura.png)


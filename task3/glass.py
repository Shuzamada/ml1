import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

glass = pd.read_csv('glass.csv')

# Удаление первого признака (Id number)
X = glass.iloc[:, 1:-1].values  # Признаки без Id
y = glass.iloc[:, -1].values    # Метки классов (Type)

# Масштабирование признаков
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Исследование зависимости ошибки от количества соседей
k_values = range(1, 31)
train_errors = []
test_errors = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    
    train_pred = knn.predict(X_train)
    test_pred = knn.predict(X_test)
    
    train_error = 1 - accuracy_score(y_train, train_pred)
    test_error = 1 - accuracy_score(y_test, test_pred)
    
    train_errors.append(train_error)
    test_errors.append(test_error)

plt.figure(figsize=(10, 6))
plt.plot(k_values, train_errors, 'o-', label='Ошибка на обучающей выборке')
plt.plot(k_values, test_errors, 'o-', label='Ошибка на тестовой выборке')
plt.xlabel('Количество соседей k')
plt.ylabel('Ошибка')
plt.title('Зависимость ошибки от количества соседей')
plt.legend()
plt.grid(True)
plt.savefig('knn_error_vs_k.png')
plt.show()

# Исследование влияния метрики расстояния
metrics = ['euclidean', 'manhattan', 'chebyshev', 'minkowski']
k_optimal = k_values[np.argmin(test_errors)] 

metric_accuracies = []
for metric in metrics:
    knn = KNeighborsClassifier(n_neighbors=k_optimal, metric=metric)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    metric_accuracies.append(accuracy)
    print(f"Метрика {metric}: точность = {accuracy:.4f}")

# Определение типа стекла для заданного экземпляра
new_sample = np.array([[1.516, 11.7, 1.01, 1.19, 72.59, 0.43, 11.44, 0.02, 0.1]])
new_sample_scaled = scaler.transform(new_sample)

best_metric = metrics[np.argmax(metric_accuracies)]
knn = KNeighborsClassifier(n_neighbors=k_optimal, metric=best_metric)
knn.fit(X_train, y_train)
predicted_type = knn.predict(new_sample_scaled)

print(f"Предсказанный тип стекла: {predicted_type[0]}")
input()


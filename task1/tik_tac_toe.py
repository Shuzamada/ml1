 import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv('tic_tac_toe.txt', header=None)
data = pd.get_dummies(data)
X = data.iloc[:, :-2].values  # Признаки
y = data.iloc[:, -1].values   # Метки классов (positive/negative)

train_sizes = np.linspace(0.1, 0.9, 9)
train_accuracies = []
test_accuracies = []

for size in train_sizes:
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=size, random_state=42)
    
    # Создание и обучение модели
    model = GaussianNB()
    model.fit(X_train, y_train)
    
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)

# Построение графика
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_accuracies, 'o-', label='Точность на обучающей выборке')
plt.plot(train_sizes, test_accuracies, 'o-', label='Точность на тестовой выборке')
plt.xlabel('Доля обучающей выборки')
plt.ylabel('Точность')
plt.title('Зависимость точности от размера обучающей выборки')
plt.legend()
plt.grid(True)
plt.savefig('accuracy_vs_trainsize_tictactoe.png')
plt.show()
input()

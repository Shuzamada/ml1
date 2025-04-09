import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.inspection import DecisionBoundaryDisplay

data = pd.read_csv('svmdata_a.txt', sep='\t')
test_data = pd.read_csv('svmdata_a_test.txt', sep='\t')

X_train = data.iloc[:, :2].values  # Признаки X1 и X2
y_train = data.iloc[:, 2].values   # Метки классов
X_test = test_data.iloc[:, :2].values
y_test = test_data.iloc[:, 2].values

# Преобразование меток классов в числовой формат
if isinstance(y_train[0], str):
    label_map = {'red': 0, 'green': 1}
    y_train = np.array([label_map[label] for label in y_train])
    y_test = np.array([label_map[label] for label in y_test])

# Создание и обучение модели SVM с линейным ядром
clf = svm.SVC(kernel='linear', C=1.0)
clf.fit(X_train, y_train)

print(f"Количество опорных векторов: {len(clf.support_vectors_)}")

# Предсказания и матрица ошибок для обучающей выборки
y_train_pred = clf.predict(X_train)
train_cm = confusion_matrix(y_train, y_train_pred)
print("Матрица ошибок на обучающей выборке:")
print(train_cm)
print(f"Точность на обучающей выборке: {accuracy_score(y_train, y_train_pred):.4f}")

# Предсказания и матрица ошибок для тестовой выборки
y_test_pred = clf.predict(X_test)
test_cm = confusion_matrix(y_test, y_test_pred)
print("Матрица ошибок на тестовой выборке:")
print(test_cm)
print(f"Точность на тестовой выборке: {accuracy_score(y_test, y_test_pred):.4f}")

# Визуализация разбиения пространства признаков
plt.figure(figsize=(10, 8))
DecisionBoundaryDisplay.from_estimator(
    clf, X_train, response_method="predict", alpha=0.5, cmap=plt.cm.coolwarm
)

# Отображение точек обучающей выборки
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=30, edgecolor='k')

# Отображение опорных векторов
plt.scatter(
    clf.support_vectors_[:, 0],
    clf.support_vectors_[:, 1],
    s=100,
    linewidth=1,
    facecolors='none',
    edgecolors='k'
)

plt.title("SVM с линейным ядром")
plt.xlabel("Признак 1")
plt.ylabel("Признак 2")
plt.savefig('svm_linear.png')
plt.show()
 

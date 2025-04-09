import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score

# Загрузка данных
data = pd.read_csv('svmdata_a.txt', sep='\t')
test_data = pd.read_csv('svmdata_a_test.txt', sep='\t')

# Подготовка данных
X_train = data.iloc[:, :2].values
y_train = data.iloc[:, 2].values
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

# Количество опорных векторов
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
plt.title("Разделение классов с помощью SVM")

# Создание сетки для визуализации границы решения
x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Отображение областей классификации
plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)

# Отображение точек обучающей выборки
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.RdYlBu, edgecolor='k')

# Отображение опорных векторов
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100, 
            linewidth=1, facecolors='none', edgecolors='k')

plt.xlabel("X1")
plt.ylabel("X2")
plt.savefig('svm_linear.png')
plt.show()

# Визуализация матрицы ошибок для обучающей выборки
plt.figure(figsize=(8, 6))
sns.heatmap(train_cm, annot=True, fmt='d', cmap='YlOrBr', 
            xticklabels=['0', '1'], yticklabels=['0', '1'])
plt.title('Матрица ошибок (обучающая выборка)')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.savefig('confusion_matrix_train.png')
plt.show()

# Визуализация матрицы ошибок для тестовой выборки
plt.figure(figsize=(8, 6))
sns.heatmap(test_cm, annot=True, fmt='d', cmap='YlOrBr', 
            xticklabels=['0', '1'], yticklabels=['0', '1'])
plt.title('Матрица ошибок (тестовая выборка)')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.savefig('confusion_matrix_test.png')
plt.show()

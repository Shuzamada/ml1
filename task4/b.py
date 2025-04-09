import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score

# Загрузка данных
data = pd.read_csv('svmdata_b.txt', sep='\t')
test_data = pd.read_csv('svmdata_b_test.txt', sep='\t')

# Подготовка данных
X_train = data.iloc[:, :2].values  # Столбцы X1 и X2 (индексы 0 и 1)
y_train = data.iloc[:, 2].values   # Столбец Colors (индекс 2)
X_test = test_data.iloc[:, :2].values
y_test = test_data.iloc[:, 2].values

# Преобразование меток классов в числовой формат
if isinstance(y_train[0], str):
    label_map = {'red': 0, 'green': 1}
    y_train = np.array([label_map[label] for label in y_train])
    y_test = np.array([label_map[label] for label in y_test])

# Перебор значений параметра C
C_values = [0.1, 1, 10, 100, 1000, 10000]
train_scores = []
test_scores = []

for C in C_values:
    clf = svm.SVC(kernel='linear', C=C)
    clf.fit(X_train, y_train)
    
    train_score = accuracy_score(y_train, clf.predict(X_train))
    test_score = accuracy_score(y_test, clf.predict(X_test))
    
    train_scores.append(train_score)
    test_scores.append(test_score)
    
    print(f"C={C}, точность на обучении: {train_score:.4f}, на тесте: {test_score:.4f}")

# Визуализация зависимости точности от параметра C
plt.figure(figsize=(10, 6))
plt.semilogx(C_values, train_scores, 'b-o', label='Обучающая выборка')
plt.semilogx(C_values, test_scores, 'r-o', label='Тестовая выборка')
plt.xlabel('Параметр C')
plt.ylabel('Точность')
plt.title('Зависимость точности от параметра C')
plt.legend()
plt.grid(True)
plt.savefig('svm_C_parameter.png')
plt.show()

# Выбор оптимального значения C
optimal_C_index = np.argmax(test_scores)
optimal_C = C_values[optimal_C_index]
print(f"Оптимальное значение C: {optimal_C}")

# Создание модели с оптимальным C
clf = svm.SVC(kernel='linear', C=optimal_C)
clf.fit(X_train, y_train)

# Предсказания и матрица ошибок для оптимальной модели
y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)
train_cm = confusion_matrix(y_train, y_train_pred)
test_cm = confusion_matrix(y_test, y_test_pred)

# Визуализация разбиения пространства признаков
plt.figure(figsize=(10, 8))
plt.title(f"Разделение классов с помощью SVM (C={optimal_C})")

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
plt.savefig('svm_optimal_C.png')
plt.show()

# Визуализация матрицы ошибок для обучающей выборки
plt.figure(figsize=(8, 6))
sns.heatmap(train_cm, annot=True, fmt='d', cmap='YlOrBr', 
            xticklabels=['0', '1'], yticklabels=['0', '1'])
plt.title(f'Матрица ошибок (обучающая выборка, C={optimal_C})')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.savefig('confusion_matrix_train_optimal_C.png')
plt.show()

# Визуализация матрицы ошибок для тестовой выборки
plt.figure(figsize=(8, 6))
sns.heatmap(test_cm, annot=True, fmt='d', cmap='YlOrBr', 
            xticklabels=['0', '1'], yticklabels=['0', '1'])
plt.title(f'Матрица ошибок (тестовая выборка, C={optimal_C})')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.savefig('confusion_matrix_test_optimal_C.png')
plt.show()

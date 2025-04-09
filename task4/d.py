import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score

# Загрузка данных
data = pd.read_csv('svmdata_d.txt', sep='\t')
test_data = pd.read_csv('svmdata_d_test.txt', sep='\t')

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

# Функция для визуализации разбиения и матрицы ошибок
def visualize_svm(clf, X_train, y_train, X_test, y_test, title, filename_prefix):
    # Предсказания
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    
    # Точность
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    # Матрицы ошибок
    train_cm = confusion_matrix(y_train, y_train_pred)
    test_cm = confusion_matrix(y_test, y_test_pred)
    
    # Визуализация разбиения
    plt.figure(figsize=(10, 8))
    plt.title(f"{title}\nТочность: обуч. {train_acc:.4f}, тест {test_acc:.4f}")
    
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
    
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.savefig(f'{filename_prefix}_boundary.png')
    plt.show()
    
    # Визуализация матрицы ошибок для обучающей выборки
    plt.figure(figsize=(8, 6))
    sns.heatmap(train_cm, annot=True, fmt='d', cmap='YlOrBr', 
                xticklabels=['0', '1'], yticklabels=['0', '1'])
    plt.title(f'Матрица ошибок (обучающая выборка)\n{title}')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.savefig(f'{filename_prefix}_cm_train.png')
    plt.show()
    
    # Визуализация матрицы ошибок для тестовой выборки
    plt.figure(figsize=(8, 6))
    sns.heatmap(test_cm, annot=True, fmt='d', cmap='YlOrBr', 
                xticklabels=['0', '1'], yticklabels=['0', '1'])
    plt.title(f'Матрица ошибок (тестовая выборка)\n{title}')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.savefig(f'{filename_prefix}_cm_test.png')
    plt.show()
    
    return train_acc, test_acc

# Полиномиальные ядра разных степеней
for degree in range(1, 6):
    clf_poly = svm.SVC(kernel='poly', degree=degree)
    clf_poly.fit(X_train, y_train)
    visualize_svm(clf_poly, X_train, y_train, X_test, y_test, 
                  f"SVM с полиномиальным ядром (степень {degree})", f"svm_poly_{degree}")

# RBF ядро
clf_rbf = svm.SVC(kernel='rbf')
clf_rbf.fit(X_train, y_train)
visualize_svm(clf_rbf, X_train, y_train, X_test, y_test, 
              "SVM с RBF ядром", "svm_rbf")

# Сигмоидальное ядро
clf_sigmoid = svm.SVC(kernel='sigmoid')
clf_sigmoid.fit(X_train, y_train)
visualize_svm(clf_sigmoid, X_train, y_train, X_test, y_test, 
              "SVM с сигмоидальным ядром", "svm_sigmoid")

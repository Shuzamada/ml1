import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

# Загрузка данных
data = pd.read_csv('svmdata_e.txt', sep='\t')
try:
    test_data = pd.read_csv('svmdata_e_test.txt', sep='\t')
    X_train = data.iloc[:, :2].values
    y_train = data.iloc[:, 2].values
    X_test = test_data.iloc[:, :2].values
    y_test = test_data.iloc[:, 2].values
except:
    # Если нет отдельного тестового файла, разделим данные
    X = data.iloc[:, :2].values
    y = data.iloc[:, 2].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Преобразование меток классов в числовой формат
if isinstance(y_train[0], str):
    label_map = {'red': 0, 'green': 1}
    y_train = np.array([label_map[label] for label in y_train])
    y_test = np.array([label_map[label] for label in y_test])

# Функция для визуализации разбиения и матрицы ошибок
def visualize_svm_gamma(kernel, gamma_values, X_train, y_train, X_test, y_test):
    plt.figure(figsize=(20, 15))
    train_accuracies = []
    test_accuracies = []
    
    for i, gamma in enumerate(gamma_values):
        # Создание и обучение модели
        clf = svm.SVC(kernel=kernel, gamma=gamma)
        clf.fit(X_train, y_train)
        
        # Предсказания
        y_train_pred = clf.predict(X_train)
        y_test_pred = clf.predict(X_test)
        
        # Точность
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        
        # Визуализация разбиения
        plt.subplot(2, 3, i+1)
        
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
        
        plt.title(f"gamma={gamma}\nТочность: обуч. {train_acc:.4f}, тест {test_acc:.4f}")
        plt.xlabel("X1")
        plt.ylabel("X2")
    
    plt.tight_layout()
    plt.savefig(f'svm_{kernel}_gamma_effect.png')
    plt.show()
    
    # График зависимости точности от gamma
    plt.figure(figsize=(10, 6))
    plt.semilogx(gamma_values, train_accuracies, 'b-o', label='Обучающая выборка')
    plt.semilogx(gamma_values, test_accuracies, 'r-o', label='Тестовая выборка')
    plt.xlabel('Параметр gamma')
    plt.ylabel('Точность')
    plt.title(f'Зависимость точности от параметра gamma ({kernel} ядро)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'svm_{kernel}_gamma_accuracy.png')
    plt.show()
    
    # Визуализация матрицы ошибок для лучшего значения gamma
    best_gamma_idx = np.argmax(test_accuracies)
    best_gamma = gamma_values[best_gamma_idx]
    
    clf = svm.SVC(kernel=kernel, gamma=best_gamma)
    clf.fit(X_train, y_train)
    
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    
    train_cm = confusion_matrix(y_train, y_train_pred)
    test_cm = confusion_matrix(y_test, y_test_pred)
    
    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    sns.heatmap(train_cm, annot=True, fmt='d', cmap='YlOrBr', 
                xticklabels=['0', '1'], yticklabels=['0', '1'])
    plt.title(f'Матрица ошибок (обучающая выборка)\n{kernel} ядро, gamma={best_gamma}')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    
    plt.subplot(1, 2, 2)
    sns.heatmap(test_cm, annot=True, fmt='d', cmap='YlOrBr', 
                xticklabels=['0', '1'], yticklabels=['0', '1'])
    plt.title(f'Матрица ошибок (тестовая выборка)\n{kernel} ядро, gamma={best_gamma}')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    
    plt.tight_layout()
    plt.savefig(f'svm_{kernel}_best_gamma_cm.png')
    plt.show()

# Значения gamma для тестирования
gamma_values = [0.001, 0.01, 0.1, 1, 10, 100]

# Тестирование для разных ядер
kernels = ['poly', 'rbf', 'sigmoid']
for kernel in kernels:
    print(f"Исследование эффекта переобучения для {kernel} ядра:")
    visualize_svm_gamma(kernel, gamma_values, X_train, y_train, X_test, y_test)

# Дополнительно: исследование влияния степени полиномиального ядра
if 'poly' in kernels:
    plt.figure(figsize=(20, 15))
    degrees = [1, 2, 3, 4, 5]
    gamma = 1.0  # Фиксированное значение gamma
    
    for i, degree in enumerate(degrees):
        clf = svm.SVC(kernel='poly', degree=degree, gamma=gamma)
        clf.fit(X_train, y_train)
        
        # Предсказания
        y_train_pred = clf.predict(X_train)
        y_test_pred = clf.predict(X_test)
        
        # Точность
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        
        # Визуализация разбиения
        plt.subplot(2, 3, i+1)
        
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
        
        plt.title(f"Полиномиальное ядро, степень {degree}\nТочность: обуч. {train_acc:.4f}, тест {test_acc:.4f}")
        plt.xlabel("X1")
        plt.ylabel("X2")
    
    plt.tight_layout()
    plt.savefig('svm_poly_degree_effect.png')
    plt.show()

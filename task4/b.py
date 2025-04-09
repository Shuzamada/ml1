import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.inspection import DecisionBoundaryDisplay

data = pd.read_csv('svmdata_b.txt', sep='\t')
test_data = pd.read_csv('svmdata_b_test.txt', sep='\t')

X_train = data.iloc[:, :2].values
y_train = data.iloc[:, 2].values
X_test = test_data.iloc[:, :2].values
y_test = test_data.iloc[:, 2].values

if isinstance(y_train[0], str):
    label_map = {'red': 0, 'green': 1}
    y_train = np.array([label_map[label] for label in y_train])
    y_test = np.array([label_map[label] for label in y_test])

C_values = [0.1, 1, 10, 100, 1000, 10000]
train_errors = []
test_errors = []

for C in C_values:
    clf = svm.SVC(kernel='linear', C=C)
    clf.fit(X_train, y_train)
    
    # Ошибки на обучающей выборке
    y_train_pred = clf.predict(X_train)
    train_error = 1 - accuracy_score(y_train, y_train_pred)
    train_errors.append(train_error)
    
    # Ошибки на тестовой выборке
    y_test_pred = clf.predict(X_test)
    test_error = 1 - accuracy_score(y_test, y_test_pred)
    test_errors.append(test_error)
    
    print(f"C={C}, ошибка на обучении: {train_error:.4f}, ошибка на тесте: {test_error:.4f}")

# Визуализация зависимости ошибок от параметра C
plt.figure(figsize=(10, 6))
plt.semilogx(C_values, train_errors, 'o-', label='Ошибка на обучающей выборке')
plt.semilogx(C_values, test_errors, 'o-', label='Ошибка на тестовой выборке')
plt.xlabel('Параметр C')
plt.ylabel('Ошибка')
plt.title('Зависимость ошибки от параметра C')
plt.legend()
plt.grid(True)
plt.savefig('svm_C_parameter.png')
plt.show()

optimal_C_index = np.argmin(test_errors)
optimal_C = C_values[optimal_C_index]
print(f"Оптимальное значение C: {optimal_C}")

clf = svm.SVC(kernel='linear', C=optimal_C)
clf.fit(X_train, y_train)

plt.figure(figsize=(10, 8))
DecisionBoundaryDisplay.from_estimator(
    clf, X_train, response_method="predict", alpha=0.5, cmap=plt.cm.coolwarm
)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=30, edgecolor='k')
plt.scatter(
    clf.support_vectors_[:, 0],
    clf.support_vectors_[:, 1],
    s=100,
    linewidth=1,
    facecolors='none',
    edgecolors='k'
)
plt.title(f"SVM с линейным ядром (C={optimal_C})")
plt.xlabel("Признак 1")
plt.ylabel("Признак 2")
plt.savefig('svm_optimal_C.png')
plt.show()

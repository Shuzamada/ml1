import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.inspection import DecisionBoundaryDisplay

data = pd.read_csv('svmdata_c.txt', sep='\t')
test_data = pd.read_csv('svmdata_c_test.txt', sep='\t')

X_train = data.iloc[:, :2].values
y_train = data.iloc[:, 2].values
X_test = test_data.iloc[:, :2].values
y_test = test_data.iloc[:, 2].values

if isinstance(y_train[0], str):
    label_map = {'red': 0, 'green': 1}
    y_train = np.array([label_map[label] for label in y_train])
    y_test = np.array([label_map[label] for label in y_test])

# Список ядер для сравнения
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
degrees = [1, 2, 3, 4, 5]  # Степени для полиномиального ядра

# Линейное ядро
plt.figure(figsize=(10, 8))
clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)
y_test_pred = clf.predict(X_test)
test_acc = accuracy_score(y_test, y_test_pred)

DecisionBoundaryDisplay.from_estimator(
    clf, X_train, response_method="predict", alpha=0.5, cmap=plt.cm.coolwarm
)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=20, edgecolor='k')
plt.title(f"Линейное ядро (точность: {test_acc:.4f})")
plt.xlabel("Признак 1")
plt.ylabel("Признак 2")
plt.savefig('svm_linear_kernel.png')
plt.show()

# Полиномиальные ядра разных степеней
plt.figure(figsize=(15, 10))
for i, degree in enumerate(degrees):
    plt.subplot(2, 3, i+1)
    clf = svm.SVC(kernel='poly', degree=degree)
    clf.fit(X_train, y_train)
    y_test_pred = clf.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    DecisionBoundaryDisplay.from_estimator(
        clf, X_train, response_method="predict", alpha=0.5, cmap=plt.cm.coolwarm
    )
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=20, edgecolor='k')
    plt.title(f"Полиномиальное ядро, степень {degree} (точность: {test_acc:.4f})")
    plt.xlabel("Признак 1")
    plt.ylabel("Признак 2")

plt.tight_layout()
plt.savefig('svm_poly_kernels.png')
plt.show()

# RBF и сигмоидальное ядра
plt.figure(figsize=(15, 6))
for i, kernel in enumerate(['rbf', 'sigmoid']):
    plt.subplot(1, 2, i+1)
    clf = svm.SVC(kernel=kernel)
    clf.fit(X_train, y_train)
    y_test_pred = clf.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    DecisionBoundaryDisplay.from_estimator(
        clf, X_train, response_method="predict", alpha=0.5, cmap=plt.cm.coolwarm
    )
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=20, edgecolor='k')
    plt.title(f"{kernel} ядро (точность: {test_acc:.4f})")
    plt.xlabel("Признак 1")
    plt.ylabel("Признак 2")

plt.tight_layout()
plt.savefig('svm_rbf_sigmoid_kernels.png')
plt.show()

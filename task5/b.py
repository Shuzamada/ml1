import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier

# Загрузка данных
data = pd.read_csv("spam7.csv")

# Преобразование целевой переменной 'yesno' в бинарный формат (0 и 1)
data['yesno'] = data['yesno'].map({'y': 1, 'n': 0})

# Разделение данных на признаки и целевую переменную
X = data.drop(columns=['yesno'])
y = data['yesno']

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Подбор оптимальной глубины дерева
best_depth = None
best_accuracy = 0
for depth in range(1, 21):
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    if acc > best_accuracy:
        best_accuracy = acc
        best_depth = depth

print(f"Оптимальная глубина дерева: {best_depth}, точность: {best_accuracy:.2f}")

# Обучение оптимального дерева
optimal_tree = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
optimal_tree.fit(X_train, y_train)

# Визуализация дерева решений
plt.figure(figsize=(30, 20))
plot_tree(optimal_tree, feature_names=X.columns, class_names=['No', 'Yes'], filled=True)
plt.show()

# Определение наиболее важных признаков
feature_importances = pd.Series(optimal_tree.feature_importances_, index=X.columns)
feature_importances = feature_importances.sort_values(ascending=False)
print("Наиболее важные признаки:")
print(feature_importances)

# Оценка качества модели
y_pred_train = optimal_tree.predict(X_train)
y_pred_test = optimal_tree.predict(X_test)
print("Точность на обучающей выборке:", accuracy_score(y_train, y_pred_train))
print("Точность на тестовой выборке:", accuracy_score(y_test, y_pred_test))
print("Отчет классификации:")
print(classification_report(y_test, y_pred_test))

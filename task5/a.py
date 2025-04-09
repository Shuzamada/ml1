import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

# Загрузка данных
df = pd.read_csv("glass.csv")

# Удаление ненужного столбца Id
df = df.drop(columns=["Id"])

# Разделение данных на признаки и целевую переменную
X = df.drop(columns=["Type"])
y = df["Type"]

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Функция для обучения и оценки дерева решений
def train_and_evaluate_tree(criterion='gini', max_depth=None):
    model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)

    # Оценка точности
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    # Визуализация дерева
    plt.figure(figsize=(32, 24))
    plot_tree(model, feature_names=X.columns, class_names=[str(c) for c in np.unique(y)], filled=True)
    plt.title(f"Decision Tree (criterion={criterion}, max_depth={max_depth})")
    plt.show()

    return train_acc, test_acc


# Исследование различных параметров
criteria = ['gini', 'entropy']
max_depths = [None, 3, 5, 10]

results = []
for criterion in criteria:
    for max_depth in max_depths:
        train_acc, test_acc = train_and_evaluate_tree(criterion, max_depth)
        results.append({"criterion": criterion, "max_depth": max_depth, "train_acc": train_acc, "test_acc": test_acc})
        print(f"Criterion: {criterion}, Max Depth: {max_depth}, Train Acc: {train_acc:.2f}, Test Acc: {test_acc:.2f}")

# Вывод результатов
results_df = pd.DataFrame(results)
print(results_df)

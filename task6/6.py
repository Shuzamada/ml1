import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix

# Загрузка данных
train_data = pd.read_csv("bank_scoring_train.csv", sep="\t")
test_data = pd.read_csv("bank_scoring_test.csv", sep="\t")  # Укажи sep="\t"

# Разделение данных на признаки и целевую переменную
X_train = train_data.drop(columns=["SeriousDlqin2yrs"])
y_train = train_data["SeriousDlqin2yrs"]
X_test = test_data.drop(columns=["SeriousDlqin2yrs"])
y_test = test_data["SeriousDlqin2yrs"]

# Заполнение пропущенных значений медианой
X_train.fillna(X_train.median(), inplace=True)
X_test.fillna(X_test.median(), inplace=True)

# Масштабирование данных
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Обучение логистической регрессии
log_reg = LogisticRegression()
log_reg.fit(X_train_scaled, y_train)
y_pred_log = log_reg.predict(X_test_scaled)
y_pred_proba_log = log_reg.predict_proba(X_test_scaled)[:, 1]

# Обучение дерева решений
decision_tree = DecisionTreeClassifier(max_depth=5, random_state=42)
decision_tree.fit(X_train, y_train)
y_pred_tree = decision_tree.predict(X_test)
y_pred_proba_tree = decision_tree.predict_proba(X_test)[:, 1]

# Оценка качества моделей
accuracy_log = accuracy_score(y_test, y_pred_log)
auc_log = roc_auc_score(y_test, y_pred_proba_log)
accuracy_tree = accuracy_score(y_test, y_pred_tree)
auc_tree = roc_auc_score(y_test, y_pred_proba_tree)

# Вывод результатов
print(f"Логистическая регрессия - Accuracy: {accuracy_log:.4f}, AUC-ROC: {auc_log:.4f}")
print(f"Дерево решений - Accuracy: {accuracy_tree:.4f}, AUC-ROC: {auc_tree:.4f}")

print("Отчет классификации - Логистическая регрессия:")
print(classification_report(y_test, y_pred_log))
print("Матрица ошибок - Логистическая регрессия:")
print(confusion_matrix(y_test, y_pred_log))

print("Отчет классификации - Дерево решений:")
print(classification_report(y_test, y_pred_tree))
print("Матрица ошибок - Дерево решений:")
print(confusion_matrix(y_test, y_pred_tree))

# Визуализация важности признаков для дерева решений
feature_importances = pd.Series(decision_tree.feature_importances_, index=X_train.columns)
plt.figure(figsize=(10, 8))
feature_importances.sort_values(ascending=False).plot(kind='bar')
plt.title("Важность признаков в дереве решений")
plt.xticks(rotation=90, fontsize=10)  # Уменьшаем размер шрифта для подписей оси X
plt.tight_layout()
plt.show()


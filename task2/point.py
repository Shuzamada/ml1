import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, precision_recall_curve, auc
from sklearn.model_selection import train_test_split

mean_class1_x1 = 0
mean_class1_x2 = 0
mean_class2_x1 = 3
mean_class2_x2 = 3
std_class1 = 1
std_class2 = 1

# Генерация данных
np.random.seed(42)
n_samples = 50

X1_class1 = np.random.normal(mean_class1_x1, std_class1, n_samples)
X2_class1 = np.random.normal(mean_class1_x2, std_class1, n_samples)
X1_class2 = np.random.normal(mean_class2_x1, std_class2, n_samples)
X2_class2 = np.random.normal(mean_class2_x2, std_class2, n_samples)

# Объединение данных
X = np.vstack([np.column_stack((X1_class1, X2_class1)), 
               np.column_stack((X1_class2, X2_class2))])
y = np.hstack([np.full(n_samples, -1), np.full(n_samples, 1)])

# Визуализация данных
plt.figure(figsize=(10, 6))
plt.scatter(X1_class1, X2_class1, color='blue', label='Класс -1')
plt.scatter(X1_class2, X2_class2, color='red', label='Класс 1')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Сгенерированные данные')
plt.legend()
plt.grid(True)
plt.savefig('generated_data.png')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = GaussianNB()
model.fit(X_train, y_train)

# Оценка качества
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Точность: {accuracy:.4f}")
print("Матрица ошибок:")
print(conf_matrix)

# Построение ROC-кривой
y_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test == 1, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC кривая (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-кривая')
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig('roc_curve.png')
plt.show()

# Построение PR-кривой
precision, recall, _ = precision_recall_curve(y_test == 1, y_prob)
pr_auc = auc(recall, precision)

plt.figure(figsize=(10, 6))
plt.plot(recall, precision, color='blue', lw=2, label=f'PR кривая (AUC = {pr_auc:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PR-кривая')
plt.legend(loc="lower left")
plt.grid(True)
plt.savefig('pr_curve.png')
plt.show()
 

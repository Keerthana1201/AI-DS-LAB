from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report

irisData = load_iris()
x = irisData.data
y = irisData.target
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)

print("ACCURACY:", accuracy_score(y_test, pred))
print("CLASSIFICATION REPORT")
print(classification_report(y_test, pred))

print("CORRECT PREDICTIONS:")
for i in range(len(y_test)):
    if y_test[i] == pred[i]:
        print(f"Index: {i}, Actual: {y_test[i]}, Predicted: {pred[i]}")

print("WRONG PREDICTIONS:")
for i in range(len(y_test)):
    if y_test[i] != pred[i]:
        print(f"Index: {i}, Actual: {y_test[i]}, Predicted: {pred[i]}")

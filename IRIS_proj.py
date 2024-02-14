from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

x,y=load_iris(return_X_y=True)


x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.2)


d = DecisionTreeClassifier()

d.fit(x_train,y_train)
y_pred = d.predict(x_test)
print(y_pred)
print(accuracy_score(y_pred,y_test)*100)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
from joblib import dump, load

df = pd.read_csv("./data/iris.csv")
print(df)
x = df.iloc[:,0:4].values
y = df.iloc[:,-1].values


#data preprocessing
#catagorie_features = ["species"]
#one_hot = OneHotEncoder()
#transformer = ColumnTransformer([("one_hot",
#                                  one_hot,
#                                  catagorie_features)],
#                                  remainder="passthrough")

transformer = LabelEncoder()
y=transformer.fit_transform(y)

#data split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.2)

#d = MLPClassifier()
d = DecisionTreeClassifier()

d.fit(x_train,y_train)
y_pred = d.predict(x_test)

print(accuracy_score(y_pred,y_test)*100)

if accuracy_score(y_pred,y_test)*100 > 99 :
    dump(d, 'abc1.joblib')

x = load('abc1.joblib')
y_pred = x.predict(x_test)
#y = load('abc.joblib')
#y_pred2 = y.predict(x_test)


print(accuracy_score(y_pred,y_test)*100)
#print(accuracy_score(y_pred2,y_test)*100)
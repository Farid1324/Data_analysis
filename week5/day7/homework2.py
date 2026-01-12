import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import train_test_split

#Load Dataset
data=load_breast_cancer(as_frame=True)
df=data.frame
print(df.head())
print(df.info())

X=df[['mean radius','mean texture','mean perimeter','mean area']]
y=df['target']

print(df.isnull().sum())
print(df.describe())

sns.pairplot(df[['mean radius','mean texture','mean perimeter',
             'mean area']])
plt.show()

X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.25,random_state=7,stratify=y)

# Step5:Scaling (fit only on training data)
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

print(X_train_scaled.mean(axis=0))
print(X_train_scaled.std(axis=0))

#Logistic Regression
log_model=LogisticRegression()
log_model.fit(X_train_scaled,y_train)

#Knn model
knn_model=KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled,y_train)

#Evaluate models
log_pred=log_model.predict(X_test_scaled)
knn_pred=knn_model.predict(X_test_scaled)


#Evaluating 
print("Logistic Regression")
print(classification_report(y_test, log_pred))
print(confusion_matrix(y_test, log_pred))

print("KNN")
print(classification_report(y_test, knn_pred))
print(confusion_matrix(y_test, knn_pred))


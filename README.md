# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries.
2.Upload and read the dataset. 
3.Check for any null values using the isnull() function. 
4.From sklearn.tree import DecisionTreeClassifier and use criterion as entropy. 
5.Find the accuracy of the model and predict the required values by importing the required module from sklearn. 
## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: VIJAY R
RegisterNumber:  212223240178
*/
```
```
import pandas as pd
data=pd.read_csv("/content/Employee.csv")
data.head()
```
![Screenshot 2024-10-16 102316](https://github.com/user-attachments/assets/a749d643-17d5-4cb9-935d-0722399ddec3)

```
data.info()
data.isnull().sum()
data["left"].value_counts()
```
![Screenshot 2024-10-16 102324](https://github.com/user-attachments/assets/0b64cf51-9389-4397-acc8-97a7b5741d22)

```
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
```
![Screenshot 2024-10-16 102341](https://github.com/user-attachments/assets/ebbb7281-0a5d-43e9-9928-3c81471dcf0f)

```
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
```
![Screenshot 2024-10-16 102349](https://github.com/user-attachments/assets/1280d39e-6520-4c4e-bcb5-05441a9fc205)

```
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```
![Screenshot 2024-10-16 102355](https://github.com/user-attachments/assets/83624ed5-2041-4680-a5ea-9da77494f315)

```
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```
![Screenshot 2024-10-16 102401](https://github.com/user-attachments/assets/88c01c0d-1dd2-490d-a336-14e8b5bf89bb)

```
import matplotlib.pyplot as plt
plt.figure(figsize=(18,6))
plot_tree(dt,feature_names=x.columns,class_names=['salary','left'],filled=True)
plt.show()
```

## Output:
![Screenshot 2024-10-16 102418](https://github.com/user-attachments/assets/b98934ea-8d4d-48a0-b686-11f24494f78a)



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.

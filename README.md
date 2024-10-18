# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. **Load Data**: Read the employee dataset.
2. **Explore Data**: Check for missing values and data types.
3. **Encode Variables**: Convert categorical features to numerical values.
4. **Define Features and Target**: Set features (`X`) and target variable (`y`).
5. **Split Data**: Split into training and testing sets (80/20).
6. **Initialize Classifier**: Create a `DecisionTreeClassifier`.
7. **Train Model**: Fit the classifier on training data.
8. **Make Predictions**: Predict outcomes on the test set.
9. **Evaluate Accuracy**: Calculate accuracy of predictions.
10. **Test New Inputs**: Predict churn for new employee data.

## Program:
```
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Ashwin Kumar A
RegisterNumber: 212223040021
```
```py
import pandas as pd
data=pd.read_csv("Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
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
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:

![image](https://github.com/user-attachments/assets/d5621a23-1907-433f-a2da-92f0d51ff77d)
![image](https://github.com/user-attachments/assets/04c2d2df-2948-4b9a-8694-2ff305790068)
![image](https://github.com/user-attachments/assets/bf6901ae-797b-45ac-9fc3-d0bad39d6381)
![image](https://github.com/user-attachments/assets/670844d6-1caa-4bb8-ab1d-ff26bf175a60)
![image](https://github.com/user-attachments/assets/d749ac2a-7c6e-4cc4-9fe0-98de8173cfcf)
![image](https://github.com/user-attachments/assets/df50933d-67ce-4d2a-8bad-f1a477d7012b)
![image](https://github.com/user-attachments/assets/ca6abf47-f134-4d8d-bca7-0eab9acffd81)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.

# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries .
2. Read the data frame using pandas.
3. Get the information regarding the null values present in the dataframe.
4. Apply label encoder to the non-numerical column inoreder to convert into numerical values.
5. Determine training and test data set.
6. Apply decision tree Classifier on to the dataframe.
7. Get the values of accuracy and data prediction.


## Program:
```
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: SOMEASVAR.R
RegisterNumber:  212221230103
```
```
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

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours",
       "time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=100)


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
## data.head():
![image](https://github.com/SOMEASVAR/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/93434149/d1197149-4dc7-480a-9a42-bea036b3f928)

## data.info():
![image](https://github.com/SOMEASVAR/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/93434149/9bdcef19-1342-4fe9-9d12-10d9f47483e7)

## isnull() and sum():
![image](https://github.com/SOMEASVAR/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/93434149/60aabbc5-2bc1-4bbd-963e-657b67c556eb)

## data value counts():
![image](https://github.com/SOMEASVAR/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/93434149/7a0a5198-3aeb-4294-82e3-67854232324a)

## data.head() for salary:
![image](https://github.com/SOMEASVAR/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/93434149/38d614ec-0e60-41c4-a997-2b491bdbc4bf)

## x.head()
:![image](https://github.com/SOMEASVAR/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/93434149/f8239c65-e574-4641-ac6f-44141a19bc43)

## Accuracy value:
![image](https://github.com/SOMEASVAR/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/93434149/0f621031-7451-45c4-8e13-f29a0e0acc35)

## Data prediction:
![image](https://github.com/SOMEASVAR/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/93434149/61002dd6-fd29-4032-bcd1-e3615e58576d)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.

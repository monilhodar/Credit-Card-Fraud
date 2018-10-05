# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#%% Importing the dataset
data=pd.read_csv('../input/creditcard.csv')
#%% print head
print(data.head())
#%% print null values
print(data.isnull().sum())
#%% eda
newdata=pd.DataFrame.copy(data)
print(newdata.head())
print(newdata.columns)
print(newdata.shape)
print(newdata.describe())
print(newdata['Class'].value_counts())
#%% 
sns.countplot(x = 'Class', data = newdata)
plt.title("Fraud class histogram")
plt.xlabel("Class")
plt.ylabel("Frequency")

#%% 
X = newdata.drop('Class', axis=1)
y = newdata['Class']
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)
y_train=y_train.astype(int)
y_test=y_test.astype(int)
# Feature Scaling 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
#%%running the model
from sklearn.linear_model import LogisticRegression
#create a model
classifier=(LogisticRegression()) 
 #fitting training data to the model
classifier.fit(X_train,y_train) 
y_pred=classifier.predict(X_test)
#%%print(list(zip(y_test, y_pred)))
#%% checking accuracy
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cm=confusion_matrix(y_test,y_pred)
print(cm) 
print("Classification report:" )
print(classification_report(y_test,y_pred))
accuracy_score=accuracy_score(y_test,y_pred)
print("Accuracy of the model: " , accuracy_score)
LogisticRegression_acc=accuracy_score

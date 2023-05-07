import numpy as np # linear algebra
import pandas as pd # 
import os
        
df=pd.read_csv('gender_classification.csv')
df.head()

Y=df['gender']
X=df.drop(['gender'],axis=1)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2)

from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
y_train = lb.fit_transform(y_train)
y_test=lb.fit_transform(y_test)

from sklearn.linear_model import LogisticRegression
model2=LogisticRegression()
model2.fit(X_train,y_train)
print(model2.score(X_test,y_test))

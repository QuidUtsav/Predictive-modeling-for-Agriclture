# All required libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load the dataset
crops = pd.read_csv("soil_measures.csv")

# Code starts from here

X=crops.drop("crop",axis=1)
y=crops["crop"]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=12)
best_predicitive_feature={}
best_feature = None
best_accuracy=0
for feature in X.columns:

    logreg = LogisticRegression(multi_class="multinomial")


    logreg.fit(X_train[[feature]],y_train)

    y_pred=logreg.predict(X_test[[feature]])
    
    accuracy = accuracy_score(y_test, y_pred)

    
    if accuracy>best_accuracy:
        best_accuracy=accuracy
        best_feature=feature

#loading the best feature and its accuracy in a dictionary

best_predicitive_feature = {
    best_feature: best_accuracy
}

print(best_predicitive_feature)

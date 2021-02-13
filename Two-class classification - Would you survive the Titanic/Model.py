import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = pd.read_csv('train_and_test2.csv')

x = data.loc[:,["Passengerid","Age","Fare","Sex","sibsp"]]
y = data.loc[:,"2urvived"]

x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=42,test_size=0.3)

classifier = LogisticRegression()
classifier.fit(x_train, y_train) 

y_predict = classifier.predict(x_test)

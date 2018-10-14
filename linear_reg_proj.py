#importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


#printing the basic info about the dataset
df=pd.read_csv("Ecommerce Customers")
print(df.head())
print(df.columns)
print(df.describe())

#some visualisation
sns.jointplot(x="Time on Website",y="Yearly Amount Spent",data=df)
plt.show()
sns.jointplot(x="Time on App",y="Yearly Amount Spent",data=df)
plt.show()
sns.jointplot(x="Time on App",y="Length of Membership",data=df,kind="hex")
plt.show()
sns.pairplot(data=df)
plt.tight_layout()
plt.show()
sns.lmplot(x="Length of Membership",y="Yearly Amount Spent",data=df)
plt.show()

#training the data model
X=df[['Avg. Session Length', 'Time on App',
       'Time on Website', 'Length of Membership']]
y=df['Yearly Amount Spent']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=101)
lm=LinearRegression()
lm.fit(X_train,y_train)
print(lm.coef_)
print(lm.intercept_)
cdf=pd.DataFrame(data=lm.coef_,index=X.columns,columns=["coeff"])
print(" Amount of time spent per unit change in different labels:")
print(cdf)
predictions=lm.predict(X_test)
plt.scatter(x=y_test,y=predictions)
plt.xlabel("original values")
plt.ylabel("predicted using model")
plt.show()


#error check
print("\nError rates : ")
print("avg error :",metrics.mean_absolute_error(y_test,predictions))
print("Square error :",metrics.mean_squared_error(y_test,predictions))
print("rms :",np.sqrt(metrics.mean_squared_error(y_test,predictions)))

sns.distplot((y_test-predictions),bins=50)
plt.show()



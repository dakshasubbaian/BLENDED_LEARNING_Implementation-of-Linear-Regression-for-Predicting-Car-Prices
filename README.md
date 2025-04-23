# BLENDED_LEARNING
# Implementation-of-Linear-Regression-for-Predicting-Car-Prices
## AIM:
To write a program to predict car prices using a linear regression model and test the assumptions for linear regression.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import Libraries: Import necessary libraries such as pandas, numpy, matplotlib, and sklearn.
2.Load Dataset: Load the dataset containing car prices and relevant features.
3.Data Preprocessing: Handle missing values and perform feature selection if necessary.
4.Split Data: Split the dataset into training and testing sets.
5.Train Model: Create a linear regression model and fit it to the training data.
6.Make Predictions: Use the model to make predictions on the test set.
7.Evaluate Model: Assess model performance using metrics like R² score, Mean Absolute Error (MAE), etc.
8.Check Assumptions: Plot residuals to check for homoscedasticity, normality, and linearity.
9.Output Results: Display the predictions and evaluation metrics.

## Program:

DEVELOPED BY: DAKSHA SUBBAIAN

REGISTER NUMBER: 212223230036
```
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import seaborn as sns

df = pd.read_csv("CarPrice_Assignment.csv")
x=df[['enginesize','horsepower','citympg','highwaympg']]
y=df['price']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
df = df.dropna()
print(df.head())
df
SS = StandardScaler()
x_trained = SS.fit_transform(X_train)
x_test = SS.fit_transform(X_test)
model = LinearRegression()
model.fit(x_trained, y_train)
y_pred=model.predict(x_test)
print("-"*50)
print("MODEL COEFFICIENTS:")
for feature, coef in zip(x.columns, model.coef_):
    print(f"(feature:>12): (coef:>10.2f)")
print(f"('Intercept':>12): (model.intercept_:>10.2f)")
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
print("\nMODEL PERFORMANCE:")
print(f"{'MSE':>12}: {mean_squared_error(y_test, y_pred):>10.2f}")
print(f"{'RMSE':>12}: {np.sqrt(mean_squared_error(y_test, y_pred)):>10.2f}")
print(f"{'R-squared':>12}: {r2_score(y_test, y_pred):>10.2f}")
print("=" * 58)
mse=mean_squared_error(y_test,y_pred)
print('Mean Squared Error=',mse)
rmse=np.sqrt(mse)
print("Root Mean Squared Error=",rmse)
r2score=r2_score(y_test,y_pred)
print('r2 score=',r2score)
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.title("Linearity Check: Actual vs Predicted Prices") 
plt.xlabel("Actual Price ($)")
plt.ylabel("Predicted Price ($)")
plt.grid(True)
plt.show()
#2.Independence (Durbin-Watson)
import statsmodels.api as sm
residuals = y_test - y_pred
dw_test = sm.stats.durbin_watson(residuals)
print(f"\nDurbin-Watson Statistic: {dw_test:.2f}",
"\n(Values close to 2 indicate no autocorrelation)")
# 3. Homoscedasticity
plt.figure(figsize=(10, 5))
sns.residplot(x=y_pred, y=residuals, lowess=True, line_kws={'color': 'red'})
plt.title("Homoscedasticity Check: Residuals vs Predicted")
plt.xlabel("Predicted Price ($)")
plt.ylabel("Residuals ($)")
plt.grid(True)
plt.show()
#4. Normality of residuals
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))
sns.histplot(residuals, kde=True, ax=ax1)
ax1.set_title("Residuals Distribution")
sm.qqplot(residuals, line='45', fit=True, ax=ax2)
ax2.set_title("Q-Q Plot")
plt.tight_layout()
plt.show()
```

## Output:
![Screenshot 2025-04-17 221615](https://github.com/user-attachments/assets/9c104f34-a71c-46c3-ab9f-c22c166e71b7)
![Screenshot 2025-04-17 221634](https://github.com/user-attachments/assets/0f52793d-3e13-44bc-9b47-a8dfa0305ed5)
![Screenshot 2025-04-17 221645](https://github.com/user-attachments/assets/77eb8347-c473-4b07-8be2-1bbbabbb58b0)
![Screenshot 2025-04-17 221703](https://github.com/user-attachments/assets/fa7faa89-1a4c-4abb-8f61-cc46b2c3a5aa)
![Screenshot 2025-04-17 221852](https://github.com/user-attachments/assets/5201475e-6325-4656-8a58-8ba85bd079f2)
![Screenshot 2025-04-17 221903](https://github.com/user-attachments/assets/3a8a1904-5525-4301-9db5-4ba0300330f9)


## Result:
Thus, the program to implement a linear regression model for predicting car prices is written and verified using Python programming, along with the testing of key assumptions for linear regression.

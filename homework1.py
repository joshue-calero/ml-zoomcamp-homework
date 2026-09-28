import numpy as np
import pandas as pd

'''Question 1'''
print("numpy version: " + np.__version__)

'''Question 2'''
print("pandas version: " +pd.__version__)

#import dataset
df = pd.read_csv(r"C:\Users\josue\Downloads\car_fuel_efficiency_2026.csv")

print(df.head(5))
print(df.columns.tolist())

asia = df[df["origin"] == "Asia"]

'''
Question 3
Records count
'''
print(f"\nRecords: {len(df)}\n")

'''
Question 4
Unique values in Fuel Type
'''
print(f"Fuel Type: {df["fuel_type"].nunique()}\n")

'''
Question 5
Missing Values
'''
print(f"Missing Values: {df.isnull().any().sum()}\n")

'''
Question 6
Max Fuel Efficiency
'''
print(f"Max fuel efficiency: {asia["fuel_efficiency_mpg"].max()}\n")

'''
Question 7
Median Value of Horsepower
'''
median = df["horsepower"].median()
mode = df["horsepower"].mode()[0]
df["horsepower"] = df["horsepower"].fillna(mode)
new_median = df["horsepower"].median()

print(f"old median: {median}, new median: {new_median}, mode: {mode}")


'''
Question 8
Sum of Weights
'''
X = asia[["vehicle_weight", "model_year"]].head(7).values

XTX = X.T @ X
XTX_inv = np.linalg.inv(XTX)

y = np.array([1100, 1300, 800, 900, 1000, 1100, 1200])

w = XTX_inv @ X.T @ y

print("Sum of weights:", w.sum())
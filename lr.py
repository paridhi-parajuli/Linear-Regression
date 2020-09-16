import pandas as pd 
import numpy as np 
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols

df_categorical=pd.read_csv('categorical.csv')
df_categorical['RegionA']=0
df_categorical.loc[df_categorical.Region == 'A', 'Region A'] = 1

print(df_categorical.head())
print(df_categorical.describe())

Y=df_categorical['Minutes']
X=df_categorical.drop(columns=['Trip','Region','Minutes'])

#LRmodel=sm.OLS(Y,sm.add_constant(X))
LRmodel = ols('Minutes ~ Parcels + TruckAge + RegionA', data=df_categorical).fit()
#result=LRmodel.fit()
print(LRmodel.summary())

fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_regress_exog(LRmodel, 'Parcels', fig=fig)
plt.show()

fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_regress_exog(LRmodel, 'TruckAge', fig=fig)
plt.show()

fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_regress_exog(LRmodel, 'RegionA', fig=fig)
plt.show()

df_cars=pd.read_csv('cars.csv')
y=df_cars['MPG']
x_disp=df_cars['Displacement']
x_cy=df_cars['Cylinders']
x=df_cars.drop(columns=['MPG','Car_Model'])
print(df_cars)

m1=sm.OLS(y,sm.add_constant(x_disp))
m1_result=m1.fit()
print(m1_result.summary())

m2=sm.OLS(y,sm.add_constant(x_cy))
m2_result=m2.fit()
print(m2_result.summary())

m3=sm.OLS(y,sm.add_constant(x))
m3_result=m3.fit()
print(m3_result.summary())

print(df_cars.corr())

















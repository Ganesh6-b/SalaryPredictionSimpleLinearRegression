# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 10:47:08 2019

@author: Ganesh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

salarydata = pd.read_csv("F:\\R\\files\\Salary_Data.csv")

salarydata.columns
salarydata = salarydata.rename(columns = {"YearsExperience": "years", "Salary" : "salary"})
salarydata.columns

plt.plot(salarydata.years, salarydata.salary)
salarydata.corr()

#build a model
import statsmodels.formula.api as smf

model1 = smf.ols("salary~years", data = salarydata).fit()

model1.summary()

model1.conf_int(0.05)
pred = model1.predict(salarydata)
pred

np.corrcoef(pred,salarydata.salary) #0.97 correlation

#2nd model

model2 = smf.ols("salary~np.log(years)", data = salarydata).fit()

model2.summary()

pred2 = model2.predict(salarydata)
np.corrcoef(pred2, salarydata.salary) #0.92 correlation

#3rd model
yearssqr = salarydata.years * salarydata.years
model3 = smf.ols("salary ~ years + yearssqr", data = salarydata).fit()

model3.summary()

pred3 = model3.predict(salarydata)
np.corrcoef(pred3, salarydata.salary) #97.8% correlation

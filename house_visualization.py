print 'import'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import operator
from sklearn.preprocessing import LabelEncoder


# import os
# directory="C:\\Users\\Youssef\\Documents\\Data_Science\\Kaggle\\House\\"
# os.chdir(directory)
# run visu_1


print ('0	reading input files')
directory="C:\\Users\\Youssef\\Documents\\Data_Science\\Kaggle\\House\\"
tr=pd.read_csv(directory+"train.csv",sep=",")
te=pd.read_csv(directory+"test.csv",sep=",")
#MSSubclass code and descriptions:
df_subclass=pd.read_csv(directory+"\inter\subclass.csv",sep=",")

print ("0	log(Price) and submission Ids")
price=np.log(tr['SalePrice'])
tr=tr.drop(['SalePrice','Id'],axis=1,inplace=False)
Id=te['Id']
te=te.drop(['Id'],axis=1,inplace=False)

print ('0	concatenate train and test sets')
dftot=pd.concat([tr,te],axis=0)
dftot.reset_index(inplace=True,drop=True)

#Columns of dftot
# [u'MSSubClass', u'MSZoning', u'LotFrontage', u'LotArea', u'Street',
       # u'Alley', u'LotShape', u'LandContour', u'Utilities', u'LotConfig',
       # u'LandSlope', u'Neighborhood', u'Condition1', u'Condition2',
       # u'BldgType', u'HouseStyle', u'OverallQual', u'OverallCond',
       # u'YearBuilt', u'YearRemodAdd', u'RoofStyle', u'RoofMatl',
       # u'Exterior1st', u'Exterior2nd', u'MasVnrType', u'MasVnrArea',
       # u'ExterQual', u'ExterCond', u'Foundation', u'BsmtQual', u'BsmtCond',
       # u'BsmtExposure', u'BsmtFinType1', u'BsmtFinSF1', u'BsmtFinType2',
       # u'BsmtFinSF2', u'BsmtUnfSF', u'TotalBsmtSF', u'Heating', u'HeatingQC',
       # u'CentralAir', u'Electrical', u'1stFlrSF', u'2ndFlrSF', u'LowQualFinSF',
       # u'GrLivArea', u'BsmtFullBath', u'BsmtHalfBath', u'FullBath',
       # u'HalfBath', u'BedroomAbvGr', u'KitchenAbvGr', u'KitchenQual',
       # u'TotRmsAbvGrd', u'Functional', u'Fireplaces', u'FireplaceQu',
       # u'GarageType', u'GarageYrBlt', u'GarageFinish', u'GarageCars',
       # u'GarageArea', u'GarageQual', u'GarageCond', u'PavedDrive',
       # u'WoodDeckSF', u'OpenPorchSF', u'EnclosedPorch', u'3SsnPorch',
       # u'ScreenPorch', u'PoolArea', u'PoolQC', u'Fence', u'MiscFeature',
       # u'MiscVal', u'MoSold', u'YrSold', u'SaleType', u'SaleCondition']


print "1	histograms"
#plotting price and log-price distributions
f,(ax1,ax2)=plt.subplots(1,2,sharey=True)
ax1.hist(np.exp(price)/100000,bins=40)
ax2.hist(price,bins=40)
ax1.set_title("Price in 100k Dollars")
ax2.set_title("log(Price)")
plt.show()



print "2	conditional scatter plot"
#Log-Price compaired to Lot Area and Dwelling type
plt.figure()
plt.scatter(np.log(tr.loc[tr["BldgType"]=="1Fam","LotArea"]),price[tr["BldgType"]=="1Fam"],alpha=0.75,color='b',label="1Fam")
plt.scatter(np.log(tr.loc[tr["BldgType"]=="2FmCon","LotArea"]),price[tr["BldgType"]=="2FmCon"],alpha=0.75,color='r',label="2FmCon")
plt.scatter(np.log(tr.loc[tr["BldgType"]=="Duplx","LotArea"]),price[tr["BldgType"]=="Duplx"],alpha=0.75,color='y',label="Duplx")
plt.scatter(np.log(tr.loc[tr["BldgType"]=="TwnhsE","LotArea"]),price[tr["BldgType"]=="TwnhsE"],alpha=0.75,color='g',label="TwnhsE")
plt.scatter(np.log(tr.loc[tr["BldgType"]=="TwnhsI","LotArea"]),price[tr["BldgType"]=="TwnhsI"],alpha=0.75,color='k',label="TwnhsI")
plt.legend(loc='upper left')
# plt.scatter(np.log(tr["LotArea"]),price,alpha=0.75,c=tr["Utilities"])
plt.axis([6,12.5,10,13.8])
plt.title("log(Price) according to LotArea and Dwelling type")
plt.show()

tr["BldgType"].groupby(by=tr["BldgType"]).count()


print "Line + std"
tab=np.random.rand(20,10)
mean=tab.mean(axis=1)
std=tab.std(axis=1)
plt.figure()
plt.errorbar(range(0,20),mean,std)
plt.show()


print "correlation between numeric features"
#First some cleaning:
#There is one irrelevant value for year built which is 2207, we replace it by the mean year of building.
dftot.loc[dftot["GarageYrBlt"]==float(2207),"GarageYrBlt"]=dftot["YearBuilt"].mode()[0]

#For GarageYearBuilt. We replace NAN by the year the house is built.
dftot.loc[dftot["GarageYrBlt"].isnull(),"GarageYrBlt"]=dftot.loc[dftot["GarageYrBlt"].isnull(),"YearBuilt"]

#For now, only the relevant column for a correlation plot are processed
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64'] 
col_corr=list(dftot.select_dtypes(include=numerics).columns)
col_corr.remove("MoSold")
col_corr.remove("YrSold")

# The NAN valeues here correspond to properties that are not applicableare 
#For example, when there is no garage, the garage area is 0 instead of NAN.
#They are all replaced by 0.
dftot[col_corr]=dftot[col_corr].fillna(value=0)


#Correlation heatmap (with seaborn)
corr = dftot[col_corr].corr().round(1)

# ---The correlation matrix is symmetric: generating a mask for the upper triangle: 
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Initializing figure
f, ax = plt.subplots(figsize=(33, 27))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3,
           square=True ,annot=True,
           linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
plt.title("Correlation matrix between features")
plt.show()

#From that map, we can see obvious and also interesting correlations:
#	- A higher area above ground matches a wider garage, a higher number of baths and rooms (but not kitchen)
#	- Recent houses have a higher garage area
#	- The features Garage area  and Garage cars are almost redundant
#	- The overall quality is highly correlated with the following features: YearBuilt, YearRemodAdd,TotalBmtSF,1stFlrSF,GrLivArea,FullBath,GarageYrBlt,Garage Area (Garage Cars too)

#GarageCar is removed for the following exploration
col_corr.remove("GarageCars")


print "correlation between numeric features and SalePrice"
#Work in progress































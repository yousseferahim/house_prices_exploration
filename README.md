# house_prices_exploration
Data visualization of the Kaggle "House Prices: advanced regression" dataset

Version 1


The objective of the competition is to predict house prices. A predictive model has been implemented in the repository "House_prices_advanced_regression". In this repository, the objective is to plot some insightfull charts with matplotlib and seaborn  libraries from Python.


One of the first step of data exploration is to look at the target variable, the house prices in the training set. In the following picture, we can see that the prices (left) are skewed (skewness=1.88). A usual trick is to get logarithm of the price (right) for a more gaussian distribution. In this situation, it is natural to perform this operation since the performance criteria of the competition is the RMSE of the logarithm of the price.

![alt tag](https://cloud.githubusercontent.com/assets/23098804/20353374/edbcbba4-ac1a-11e6-908c-162ebb466169.png)


A scatter plot between between log-price and Lot area displays a clear positive correlation between those two variable (obvious). By adding the dwelling type to this scatter (color of the points), despite the unbalance between types of houses, we can say that end unit town houses (green) are more expensive than detached family houses (blue) for a similar lot area.

![alt tag](https://cloud.githubusercontent.com/assets/23098804/20353390/f660cc64-ac1a-11e6-88a1-4872c8561225.png)


Now let's focus on the relation between features. We start with numeric features. The following correlation matrix highlights some relatins between columns:
* A higher area above ground matches a wider garage, a higher number of baths and rooms (but not kitchen)
* Recent houses have a higher garage area
* The features Garage area  and Garage cars are almost redundant
* The overall quality is highly correlated with the following features: YearBuilt, YearRemodAdd,TotalBmtSF,1stFlrSF,GrLivArea,FullBath,GarageYrBlt,Garage Area (Garage Cars too)
	
![alt tag](https://cloud.githubusercontent.com/assets/23098804/20353396/fb8a2898-ac1a-11e6-8b27-9238f8cf0ea7.png)


Work in progress

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

df = pd.read_csv("data/yield_df.csv")
# print(df.head())
# print(df.shape)   100 row 7 column

df.drop('Unnamed: 0',axis=1,inplace=True)
# print(df.head())
# print(df.shape) 100 row 8 column


# findNulldatas = df.isnull().sum()   
# print(findNulldatas)      we had already full data every each row


# info = df.info()
# print(info)

# duplicatedDatas = df.duplicated().sum()  we findet duplicated datas and replace it or remove on currently datas 
# print(duplicatedDatas)


#*********************REPLACE CURRENTLY DUPLICATE DATAS

# print(df.shape)  100,10
# CurrentDuplicateDatas = df.duplicated().sum() 
# print(CurrentDuplicateDatas)  20
# DropDuplicatedData =  df.drop_duplicates(inplace=True) 
# print(df.shape) 80,10
df.drop_duplicates(inplace=True)

# Desc = df.describe()
# print(Desc)

# Correlation = df.corr()   we have to understand every datas (1 is positiv so it means datas together) or (0 means no corr= no relation) or(-1 is datas one by increase and other side decrease growth = opposite direction)  
# print(Correlation) 


# AverageRainFall = df['average_rain_fall_mm_per_year']
# print(AverageRainFall)


# def isStr(obj): # if it is string True, so its not convert float and False
#     try:
#         float(obj)
#         return False
#     except:
#         return True
    
# print(df.info())
# to_drop = df[df['average_rain_fall_mm_per_year'].apply(isStr)].index
# print(df.info())

# plt.figure(figsize=(20,20))
# sns.countplot(y=df['Area'])
# plt.show()

df["average_rain_fall_mm_per_year"] = df["average_rain_fall_mm_per_year"].astype(np.float64) #convert selected line(average_rain) to float64



# print(len(df['Area']))  #all datas from Area
# print(len(df["Area"].unique())) #unique datas from Area
# print(len(df["average_rain_fall_mm_per_year"])) 



#PRINT ALL OF STATE 
# country = df["Area"].unique() 
# for state in country:
#     print(state)


#SUM OF ALL COUNTRY RANYIELD
# print(df["hg/ha_yield"].sum())



#**** IN CURRENT COUNTRY TOTAL OF YEARLY YIELD

# yield_per_country = []
# country = df["Area"].unique()
# for state in country:
#     yield_per_country.append(df[df["Area"]==state]["hg/ha_yield"].sum())  #append in yield arr current state total yearly yield
    

# plt.figure(figsize=(10,20))
# sns.barplot(y= country, x=yield_per_country)
# plt.show()



#***** PRODUCT GRAPH

# item_products = df["Item"].value_counts() ## indicate that line how much item harvest in one year?(20patato, 10wheat, 20rice)
# print(item_products)
# sns.countplot(x=df["Item"])
# plt.show()


#****** YIELD VS ITEM 

# crops = df["Item"].unique()
# yield_per_item = []

# for crop in crops:
#     yield_per_item.append(df[df["Item"]==crop]["hg/ha_yield"].sum()) 

# sns.barplot(y=yield_per_item, x=crops)
# plt.show()




#**** Train and Test Columns


col = ["Area",   "Item",  "Year", "hg/ha_yield",  "average_rain_fall_mm_per_year",  "pesticides_tonnes",  "avg_temp"]
df = df[col]


X = df.drop('hg/ha_yield',axis=1)
y= df["hg/ha_yield"]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

# print(df.shape)  #(25932r, 7c)

# print(X_train.shape)          #(20745r, 6c) %80
# print(X_test.shape)           #(5187r, 6c) %20
# print(y_train.shape)          #(20745,) %80
# print(y_test.shape)           #(5187,) %20



#***** Convert Categorical to Numerical and Scale the Values

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

ohe = OneHotEncoder(drop='first')
scaler = StandardScaler()

#Area, Item,  Year,  hg/ha_yield , average_rain_fall_mm_per_year,  pesticides_tonnes,  avg_temp
preprocessor = ColumnTransformer(transformers=[
    ('onehotencoder', ohe,[0,1]),
    ('standrization', scaler,[2,3,4,5] )
],
    remainder='passthrough'
)


X_train_dummy =  preprocessor.fit_transform(X_train)
X_test_dummy = preprocessor.fit_transform(X_test)




# TRAIN MODEL 

from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# models = {
#     'lr' : LinearRegression(),
#     'lss' : Lasso(),
#     'rg' : Ridge(),
#     'Knr' : KNeighborsRegressor(),  #best practice
#     'dtr' : DecisionTreeRegressor(),
# }


# for name, mod in models.items():
#    mod.fit(X_train_dummy, y_train)
#    y_pred  = mod.predict(X_test_dummy)
#    print(f"{name} MSE : {mean_squared_error(y_test, y_pred)} Score : {r2_score(y_test, y_pred)}")
   
   

# SELECT MODEL 

dtr =DecisionTreeRegressor()
dtr.fit(X_train_dummy, y_train)
dtr.predict(X_test_dummy)


## Predictive System 
def prediction(Area,Item,Year,average_rain_fall_mm_per_year,pesticides_tonnes,avg_temp): 
    features = np.array([[Area,Item,Year,average_rain_fall_mm_per_year,pesticides_tonnes,avg_temp]])
    transformed_features = preprocessor.transform(features)
    predicted_value =  dtr.predict(transformed_features).reshape(1,-1)
    return predicted_value[0]
    

Year= 1991
average_rain_fall_mm_per_year=1485.0
pesticides_tonnes= 121.0
avg_temp=15.36
Area = "Albania"
Item ="Soybeans" 

result = prediction(Area,Item,Year,average_rain_fall_mm_per_year,pesticides_tonnes,avg_temp)
# Albania/Maize/1990/36613/1485.0/121.0/16.37
print(result)
print(df.head(20))


import pickle 

pickle.dump(dtr,open('dtr.pkl', 'wb'))
pickle.dump(preprocessor,('preprocessor.pkl','wb'))
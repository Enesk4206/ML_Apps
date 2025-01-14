import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler    #feature scaling
from sklearn.linear_model import LogisticRegression # trainin model
from sklearn.metrics import accuracy_score  #accuracy score
import pickle 

breast = pd.read_csv("data/data.csv")

# print(breast.head())

# valueOfDiagnosis =  breast['diagnosis'].value_counts()
# print(valueOfDiagnosis)

# shape = breast.shape
# print(shape)    569 rows 33 columns

# breastIsNull = breast.isnull().sum()
# print(breastIsNull)     0 null data we have full the fills

# DetectDuplicateDatas = breast.duplicated().sum()
# print(breastDatasIsUnıque)      All data is unique that is probably 0 


# breastInfo = breast.info()
# print(breastInfo)


# breastCorrelation = breast.corr()   1 is positiv relation || 0 is no relation || -1 negativ relation
# print(breastCorrelation)

# print(breast.describe())
# dropNanDatasFromBreast = breast.drop('Unnamed: 32',axis=1,inplace=True)     delete 1 column and inplace that 
# print(breast.describe())


# print(breast.head())
# print(breast['diagnosis'].value_counts())
# breast['diagnosis']  = breast['diagnosis'].map({'M':1, 'B':0})
# print(breast['diagnosis'].value_counts())



# *******Splitting Data into Traning and Testing Sets

shape = breast.shape
# print(shape)        #569, 33

#drop Nan values
breast.drop('Unnamed: 32',axis=1,inplace=True)


breast['diagnosis']  = breast['diagnosis'].map({'M':1, 'B':0})
X = breast.drop('diagnosis',axis=1)
# print(X.shape)          #569, 32
y = breast['diagnosis']
# print(y.shape)    #569 ,0

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2 , random_state=42)

# print(shape)    #569 ROWS, 33 COLUMNS
# print(X_train.shape)    #e ROWS, 32 COLUMNS 0.8
# print(X_test.shape)     #e ROWS, 32 COLUMNS 0.2
# print(y_train.shape)    # e ROWS 0 COLUMN    LABEL 1AXIS 0.8
# print(y_test.shape)     #e ROWS 0 COLUMN      LABEL 1AXIS 0.2


# ***Feature Scaling

sc = StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)
# print(X_train)


#****  Training Model

lg = LogisticRegression()
lg.fit(X_train, y_train)
y_pred =  lg.predict(X_test)

# 0.9736842105263158
print(accuracy_score(y_test, y_pred))

print(X_train[120])
# Prediction System 
input_text = ( -0.23711093, -0.4976419 ,  0.61365274, -0.49813131, -0.53102815,
       -0.57694824, -0.17494424, -0.36215622, -0.284859  ,  0.43345165,
        0.17818232, -0.36844966,  0.55310406, -0.31671104, -0.40524636,
        0.04025752, -0.03795529, -0.18043065,  0.16478901, -0.12170969,
        0.23079329, -0.50044002,  0.81940367, -0.46922838, -0.53308833,
       -0.04910117, -0.04160193, -0.14913653,  0.09681787,  0.10617647,
        0.49035329      

) #user input

# 913512,Cancrous,11.68,16.17,75.49,420.5,0.1128,0.09263,0.04279,0.03132,0.1853,0.06401,0.3713,1.154,2.554,27.57,0.008998,0.01292,0.01851,0.01167,0.02152,0.003213,13.32,21.59,86.57,549.8,0.1526,0.1477,0.149,0.09815,0.2804,0.08024

# 914062,M,18.01,20.56,118.4,1007,0.1001,0.1289,0.117,0.07762,0.2116,0.06077,0.7548,1.288,5.353,89.74,0.007997,0.027,0.03737,0.01648,0.02897,0.003996,21.53,26.06,143.4,1426,0.1309,0.2327,0.2544,0.1489,0.3251,0.07625


np_df = np.asarray(input_text)   #convert array
predictiont = lg.predict(np_df.reshape(1,-1))   #implement row vector one row and automatic column

if(predictiont[0] ==1):
    print("Cancrous",input_text)
else:
    print("Not Cancrous",input_text)
    
    
pickle.dump(lg, open('model.pkl', 'wb'))
<H3>NAME: SREEKUMAR S</H3>
<H3>REG.NO: 212223240157</H3>
<H3>EX. NO.1</H3>
<H3>DATE: 23.08.2024</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle:**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values. Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing:**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, the Random Forest algorithm does not support null values, therefore to execute a random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm is executed in one data set, and the best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```
from google.colab import files
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
data=pd.read_csv("/content/Churn_Modelling.csv",encoding='latin1')
print(data)
data.isnull().sum()
data.info()
data = data.drop(['Surname', 'Geography','Gender'], axis=1)
x=data.iloc[:,:-1].values
print(x)
y=data.iloc[:,-1].values
print(y)
data.duplicated().sum()
scaler = StandardScaler()
data1 = scaler.fit_transform(data)
print(data1)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print(X_train)
print(len(X_train))
print(X_test)
print(len(X_test))
```


## OUTPUT:
## DATASET:
![image](https://github.com/user-attachments/assets/ca3fe411-9163-4382-a44f-d8db03000642)
# NULL VALUES:
![image](https://github.com/user-attachments/assets/09c8b0d7-365f-4a8f-a127-7b7e8f82e970)
# DATASET INFO:
![image](https://github.com/user-attachments/assets/e6b0f1e1-fdfe-4ffe-84da-8bdb8a274240)
# X AND Y VALUES:
![image](https://github.com/user-attachments/assets/12b08ca8-f5f2-4184-9c9a-c0d380f5bdce)
# DUPLICATES:
![image](https://github.com/user-attachments/assets/9fe7e3cb-74e1-430b-9854-ae7df44807f7)
# STANDARDIZES VALUES:
![image](https://github.com/user-attachments/assets/9acce427-3b2b-415f-a556-1fb0d18a9d81)
# X_TRAIN VALUES:
![image](https://github.com/user-attachments/assets/3a88a44e-d2f8-468a-b39d-61be9d9063e8)
# X_TEST VALUES:
![image](https://github.com/user-attachments/assets/fe76b18d-757f-4f73-907a-3725af2bbd09)










## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.



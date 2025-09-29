# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
```
```
data=pd.read_csv("income(1) (1).csv",na_values=[ " ?"])
data
```
<img width="985" height="603" alt="image" src="https://github.com/user-attachments/assets/2a8c3d39-0a0d-4c50-b489-5d5d1ecf5330" />

```
 data.isnull().sum(
```
<img width="257" height="250" alt="image" src="https://github.com/user-attachments/assets/68d4cd23-83b9-4ea7-bea4-ee36ec321c8d" />

```
missing=data[data.isnull().any(axis=1)]
 missing
```
<img width="985" height="586" alt="image" src="https://github.com/user-attachments/assets/a7dcd80e-bd0b-463a-b7b9-afd57373ec3e" />

```
data2=data.dropna(axis=0)
 data2
```
<img width="1018" height="611" alt="image" src="https://github.com/user-attachments/assets/ee80736f-3744-4e0c-8dc7-4ee78c6aed0b" />

```
data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
 print(data2['SalStat'])
```
<img width="618" height="220" alt="image" src="https://github.com/user-attachments/assets/917fe5ad-f86c-4b85-bb44-8ddd826eae44" />

```
 sal2=data2['SalStat']
 dfs=pd.concat([sal,sal2],axis=1)
 dfs
```
<img width="318" height="357" alt="image" src="https://github.com/user-attachments/assets/d189b8f7-0f10-4c8c-b122-ffd62e50943b" />

```
 data2
```
<img width="986" height="459" alt="image" src="https://github.com/user-attachments/assets/3b7dd483-eb65-4243-a97c-05fecd2341a1" />

```
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```
<img width="991" height="365" alt="image" src="https://github.com/user-attachments/assets/27b5b394-b5ce-4f20-9897-172610e435ea" />

```
 columns_list=list(new_data.columns)
 print(columns_list)
```
<img width="980" height="335" alt="image" src="https://github.com/user-attachments/assets/5392739f-80b1-499e-bb6b-c55047fd1d54" />

```
features=list(set(columns_list)-set(['SalStat']))
 print(features)
```
<img width="979" height="338" alt="image" src="https://github.com/user-attachments/assets/e83a30c5-351a-4b36-856e-f89361dfcc32" />

```
 y=new_data['SalStat'].values
 print(y)
```
<img width="209" height="33" alt="image" src="https://github.com/user-attachments/assets/46a27c32-868f-44c8-8f59-6e65e56f96c0" />

```
 x=new_data[features].values
 print(x)
```
<img width="205" height="143" alt="image" src="https://github.com/user-attachments/assets/78804c13-945a-42af-8f18-cbc7103032eb" />

```
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors = 5)
KNN_classifier.fit(train_x,train_y)
```
<img width="353" height="98" alt="image" src="https://github.com/user-attachments/assets/6efa31d4-0702-4c5a-b77b-b17af0430b5a" />

``` 
 prediction=KNN_classifier.predict(test_x)
 confusionMatrix=confusion_matrix(test_y, prediction)
 print(confusionMatrix)
```
<img width="390" height="67" alt="image" src="https://github.com/user-attachments/assets/a9c49573-6ebe-431a-bd67-ad3d1d0f8386" />

```
accuracy_score=accuracy_score(test_y,prediction)
 print(accuracy_score)
```
<img width="480" height="32" alt="image" src="https://github.com/user-attachments/assets/23cb5790-750f-4a7f-b8b7-42c5e59eb8cc" />

```
 print("Misclassified Samples : %d" % (test_y !=prediction).sum())
```
<img width="494" height="42" alt="image" src="https://github.com/user-attachments/assets/4b7ab586-dd3a-422a-bb62-8b628bcd0268" />

```
data.shape
```
<img width="437" height="50" alt="image" src="https://github.com/user-attachments/assets/e7786000-a694-4758-abff-d915648f359b" />

```
 import pandas as pd
 from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
 data={
 'Feature1': [1,2,3,4,5],
 'Feature2': ['A','B','C','A','B'],
 'Feature3': [0,1,1,0,1],
 'Target'  : [0,1,1,0,1]
 }
 df=pd.DataFrame(data)
 x=df[['Feature1','Feature3']]
 y=df[['Target']]
 selector=SelectKBest(score_func=mutual_info_classif,k=1)
 x_new=selector.fit_transform(x,y)
 selected_feature_indices=selector.get_support(indices=True)
 selected_features=x.columns[selected_feature_indices]
 print("Selected Features:")
 print(selected_features)
```
<img width="720" height="52" alt="image" src="https://github.com/user-attachments/assets/0c3b27ce-578d-4aab-89c9-9736536caef5" />

```
 import pandas as pd
 import numpy as np
 from scipy.stats import chi2_contingency
 import seaborn as sns
 tips=sns.load_dataset('tips')
 tips.head()
```
<img width="627" height="178" alt="image" src="https://github.com/user-attachments/assets/f3f7a225-f2ae-4f34-b0c0-52ef1d9455ed" />

```
 tips.time.unique()
```
<img width="578" height="63" alt="image" src="https://github.com/user-attachments/assets/bb774662-1084-4c71-a9c4-e2c07bb3cd06" />

```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```
<img width="489" height="88" alt="image" src="https://github.com/user-attachments/assets/9381c8e0-ec78-4971-b535-00fa453af1e7" />

```
chi2,p,_,_=chi2_contingency(contingency_table)
 print(f"Chi-Square Statistics: {chi2}")
 print(f"P-Value: {p}")
```
<img width="554" height="55" alt="image" src="https://github.com/user-attachments/assets/ff6a4788-1737-4160-a1da-2c02f54dd5bd" />
       
# RESULT:
Thus the program to read the given data and perform Feature Scaling and Feature Selection process and save the data to a file is been executed

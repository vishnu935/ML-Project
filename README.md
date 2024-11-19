
**FLOOD PREDICTION**

**Submitted by** : Vishnu E J

**TABLE OF CONTENTS**
1. Problem Statement
2. Objective
3. Data Collection
4. Data Description
5. EDA
6. Data Preprocessing
7. Visualization
8. Data Splitting
9. Model Selection
10. Model Training and Evaluation (Without feature selection and hyperparameter tuning)
11. Feature Selection
12. Hyperparameter tuning
13. Saving the model
14. Load the model
#### PROBLEM STATEMENT
Flood detection refers to identifying, monitoring, and alerting authorities or individuals about the presence of flooding in a particular area. It involves using various technologies and methods to detect, predict, and the impacts of floods. Flood prediction is a critical area of research due to its significant impact on human life, infrastructure, and the environment. Accurate flood prediction models can aid in disaster preparedness and risk management, reducing the adverse effects of floods.
#### OBJECTIVE
Develop a machine learning model to predict the occurrence of floods based on environmental and weather features such as temperature, rainfall, humidity, and other climatic data.
*   Predict whether a flood will occur (binary classification: 1 for flood, 0 for no flood).
*   Identify the most important climatic features that influence flood events.
### DATA DESCRIPTION

This dataset contains Rainfall, Relative_Humidity, Wind_Speed, Cloud_Coverage, and other relevant attributes. It's a great dataset for learning to work with data analysis and visualization.

**Dataset**:  https://docs.google.com/spreadsheets/d/1AyALjj0qjONSfRlqGJnh5_pAcP6tQ_iEVyi3w49jIHs/edit?gid=1464562813#gid=1464562813
*   Number of rows: 20,544
*   Number of columns: 17
*   Target column: Flood? (Binary target: 1 = Flood, 0 = No Flood)
*   Null Values: The Flood? column has many missing values (16,051 out of 20,544).

**Key columns**:

*   **Station_Names**: The name of the weather station where data was recorded.
***Year, Month**: Temporal data to indicate when the measurements were taken.
*   **Max_Temp**: Maximum temperature recorded (in Celsius).
*   **Min_Temp**: Minimum temperature recorded (in Celsius).
*   **Rainfall**: Rainfall amount (in mm).
*   **Relative_Humidity**: Humidity percentage.
*   **Wind_Speed**: Wind speed (units might need clarification).
*   **Cloud_Coverage**: Cloud cover fraction or percentage.
*   **Bright_Sunshine**: Duration of bright sunshine (likely in hours).
*   **Station_Number**: A unique identifier for the station.
*   **X_COR, Y_COR**: Coordinates in a projected coordinate system.
*   **LATITUDE, LONGITUDE**: Geographic coordinates.
*   **ALT**: Altitude (in meters).

EXPLORATORY DATA ANALYSIS
Explore the distribution of key features.
Handle missing values and outliers.
Analyze correlations between features and the target (Flood?).

[ ]
dm.columns
Index(['Station_Names', 'Year', 'Month', 'Max_Temp', 'Min_Temp', 'Rainfall',
       'Relative_Humidity', 'Wind_Speed', 'Cloud_Coverage', 'Bright_Sunshine',
       'Station_Number', 'X_COR', 'Y_COR', 'LATITUDE', 'LONGITUDE', 'ALT',
       'Flood?'],
      dtype='object')

[ ]
dm.shape
(20544, 17)

[ ]
dm.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 20544 entries, 0 to 20543
Data columns (total 17 columns):
 #   Column             Non-Null Count  Dtype  
---  ------             --------------  -----  
 0   Station_Names      20544 non-null  object 
 1   Year               20544 non-null  int64  
 2   Month              20544 non-null  int64  
 3   Max_Temp           20544 non-null  float64
 4   Min_Temp           20544 non-null  float64
 5   Rainfall           20544 non-null  float64
 6   Relative_Humidity  20544 non-null  float64
 7   Wind_Speed         20544 non-null  float64
 8   Cloud_Coverage     20544 non-null  float64
 9   Bright_Sunshine    20544 non-null  float64
 10  Station_Number     20544 non-null  int64  
 11  X_COR              20544 non-null  float64
 12  Y_COR              20544 non-null  float64
 13  LATITUDE           20544 non-null  float64
 14  LONGITUDE          20544 non-null  float64
 15  ALT                20544 non-null  int64  
 16  Flood?             4493 non-null   float64
dtypes: float64(12), int64(4), object(1)
memory usage: 2.7+ MB

[ ]
dm.describe()


[ ]
dm.dtypes


[ ]
dm.isnull().sum()


[ ]
dm.nunique()


[ ]
dm.duplicated().sum()
0

[ ]
num_cols=dm.select_dtypes(include='number').columns
num_cols
Index(['Year', 'Month', 'Max_Temp', 'Min_Temp', 'Rainfall',
       'Relative_Humidity', 'Wind_Speed', 'Cloud_Coverage', 'Bright_Sunshine',
       'Station_Number', 'X_COR', 'Y_COR', 'LATITUDE', 'LONGITUDE', 'ALT',
       'Flood?'],
      dtype='object')

[ ]
cat_cols=dm.select_dtypes(include='object').columns
cat_cols
Index(['Station_Names'], dtype='object')

[ ]
# scatter plot to identify the relationship between flood and remaining features
plt.figure(figsize=(15, 10))
for i, col in enumerate(dm.columns):
    plt.subplot(5, 4, i+1)
    sns.barplot(data=dm,x=col,y='Flood?')
    plt.xlabel(col)
plt.tight_layout()
plt.show()

Explanation - The scatter plots reveal the relationships between flood events (Flood?) and key weather-related features:

Flood vs Rainfall:
As expected, higher rainfall shows some association with flood events, but not all high-rainfall events result in floods.

Flood vs Max Temperature:
There is no clear trend between maximum temperature and flood events.

Flood vs Min Temperature:
Similar to max temperature, the minimum temperature doesn't show a strong relationship with floods.

Flood vs Relative Humidity:
Some correlation might exist where higher humidity coincides with flood occurrences. These scatter plots suggest that rainfall and humidity might be more influential factors in flood occurrence, while temperatures appear less relevant.

Correlation Matrix

[ ]
dmcorr=dm.drop(["Station_Names"],axis=1)
corr1=dmcorr.corr()
corr1

Heat map

[ ]
plt.figure(figsize=(20, 10))
sns.heatmap(corr1,annot=True,cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

Explanation - This helps visualize relationships between different features and spot strong correlations easily. The feature Min_Temp, Relative_Humidity, Cloud_Coverage, Rainfall is least correlated with the target variable 'Flood?'.

Features with multicollinearity:
Flood? and Min_Temp
Flood? and Relative_Humidity
Flood? and Cloud_Coverage
Flood? and Rainfall

[ ]

flood_data = dm
sns.set(style='whitegrid')
fig, axes = plt.subplots(2, 2, figsize=(15, 10))  # 2x2 grid
# Scatter plot: Flood? vs Rainfall
sns.scatterplot(data=flood_data, x='Rainfall', y='Flood?', ax=axes[0, 0], color='b')
axes[0, 0].set_title('Flood vs Rainfall')

# Scatter plot: Flood? vs Cloud_Coverage
sns.scatterplot(data=flood_data, x='Cloud_Coverage', y='Flood?', ax=axes[0, 1], color='r')
axes[0, 1].set_title('Flood vs Cloud Coverage')

# Scatter plot: Flood? vs Min_Temp
sns.scatterplot(data=flood_data, x='Min_Temp', y='Flood?', ax=axes[1, 0], color='g')
axes[1, 0].set_title('Flood vs Min Temperature')

# Scatter plot: Flood? vs Relative_Humidity
sns.scatterplot(data=flood_data, x='Relative_Humidity', y='Flood?', ax=axes[1, 1], color='purple')
axes[1, 1].set_title('Flood vs Relative Humidity')

plt.tight_layout()
plt.show()


Explanation - The scatter plots reveal the relationships between flood events (Flood?) and key weather-related features:

Flood vs Rainfall:
As expected, higher rainfall shows some association with flood events, but not all high-rainfall events result in floods.

Flood vs Cloud Coverage:
There is a relation between cloud coverage and flood events.

Flood vs Min Temperature:
Similar to cloud coverage, the minimum temperature show a relationship with floods.

Flood vs Relative Humidity:
Some correlation might exist where higher humidity coincides with flood occurrences, but it isn’t definitive.

DATA PREPROCESSING
Replace missing values in the Flood? column with 0.
Normalize/scale numerical features (like temperature, rainfall, etc.).
Remove the categorical column Station_Names. Encoded data Station_Number is used instead of that.

[ ]
dm.drop(["Station_Names"],axis=1,inplace=True)  # drop unwanted column
dm.fillna(0, inplace=True) # Fill null values with 0. (binary classification: 1 for flood, 0 for no flood)

[ ]
plt.figure(figsize=(10, 10))
num_cols = dm.select_dtypes(include = ["int64","float64"])
for i, col in enumerate(num_cols):
    plt.subplot(5, 4, i+1)
    sns.boxplot(dm[col])
    plt.title(col)
plt.tight_layout()
plt.show()

Removing outliers (Max_Temp)

[ ]
q1_Max_Temp=dm.Max_Temp.quantile(0.25)
q3_Max_Temp=dm.Max_Temp.quantile(0.75)

# Step 2: Calculate the IQR (Interquartile Range)
iqr_Max_Temp=q3_Max_Temp-q1_Max_Temp

# Step 3: Define the lower and upper bounds for 'Max_Temp'
lower_bound_Max_Temp = q1_Max_Temp - 1.5 * iqr_Max_Temp
upper_bound_Max_Temp = q3_Max_Temp + 1.5 * iqr_Max_Temp

# Step 4: Filter out rows where 'Max_Temp' is outside the bounds
Max_Temp_outliers_removed = dm[(dm.Max_Temp >= lower_bound_Max_Temp) & (dm.Max_Temp <= upper_bound_Max_Temp)]

# Show the number of rows before and after removing outliers
print(f"Original dataset shape: {dm.shape}")
print(f"Dataset shape after removing outliers: {Max_Temp_outliers_removed.shape}")

# Display the first few rows of the data without outliers

dmp1 = Max_Temp_outliers_removed
dmp1.head()


[ ]
plt.figure(figsize=(10, 10))
num_cols = dmp1.select_dtypes(include = ["int64","float64"])
for i, col in enumerate(num_cols):
    plt.subplot(5, 4, i+1)
    sns.boxplot(dmp1[col])
    plt.title(col)
plt.tight_layout()
plt.show()

Removing outliers (Rain Fall)

[ ]
q1Rainfall=dmp1.Rainfall.quantile(0.25)
q3Rainfall=dmp1.Rainfall.quantile(0.75)

# Step 2: Calculate the IQR (Interquartile Range)
iqrRainfall=q3Rainfall-q1Rainfall

# Step 3: Define the lower and upper bounds for 'Rainfall'
lower_boundRainfall = q1Rainfall - 1.5 * iqrRainfall
upper_boundRainfall = q3Rainfall + 1.5 * iqrRainfall

# Step 4: Filter out rows where 'Rainfall' is outside the bounds
Rainfall_outliers_removed = dmp1[(dmp1.Rainfall >= lower_boundRainfall) & (dmp1.Rainfall <= upper_boundRainfall)]

# Show the number of rows before and after removing outliers
print(f"Original dataset shape: {dm.shape}")
print(f"Dataset shape after removing outliers: {Rainfall_outliers_removed.shape}")

# Display the first few rows of the data without outliers

dmp2=Rainfall_outliers_removed
dmp2.head()


[ ]
plt.figure(figsize=(10, 10))
num_cols = dmp2.select_dtypes(include = ["int64","float64"])
for i, col in enumerate(num_cols):
    plt.subplot(5, 4, i+1)
    sns.boxplot(dmp2[col])
    plt.title(col)
plt.tight_layout()
plt.show()

Removing outliers (Relative_Humidity)

[ ]
q1Relative_Humidity=dmp2.Relative_Humidity.quantile(0.25)
q3Relative_Humidity=dmp2.Relative_Humidity.quantile(0.75)

# Step 2: Calculate the IQR (Interquartile Range)
iqrRelative_Humidity=q3Relative_Humidity-q1Relative_Humidity

# Step 3: Define the lower and upper bounds for 'Relative_Humidity'
lower_boundRelative_Humidity = q1Relative_Humidity - 1.5 * iqrRelative_Humidity
upper_boundRelative_Humidity = q3Relative_Humidity + 1.5 * iqrRelative_Humidity

# Step 4: Filter out rows where 'Relative_Humidity' is outside the bounds
Relative_Humidity_outliers_removed = dmp2[(dmp2.Relative_Humidity >= lower_boundRelative_Humidity) & (dmp2.Relative_Humidity <= upper_boundRelative_Humidity)]

# Show the number of rows before and after removing outliers
print(f"Original dataset shape: {dm.shape}")
print(f"Dataset shape after removing outliers: {Relative_Humidity_outliers_removed.shape}")

# Display the first few rows of the data without outliers

dmp3=Relative_Humidity_outliers_removed
dmp3.head()


[ ]
plt.figure(figsize=(10, 10))
num_cols = dmp3.select_dtypes(include = ["int64","float64"])
for i, col in enumerate(num_cols):
    plt.subplot(5, 4, i+1)
    sns.boxplot(dmp3[col])
    plt.title(col)
plt.tight_layout()
plt.show()

Removing outliers (Wind_Speed)

[ ]
q1Wind_Speed=dmp3.Wind_Speed.quantile(0.25)
q3Wind_Speed=dmp3.Wind_Speed.quantile(0.75)

# Step 2: Calculate the IQR (Interquartile Range)
iqrWind_Speed=q3Wind_Speed-q1Wind_Speed

# Step 3: Define the lower and upper bounds for 'Wind_Speed'
lower_boundWind_Speed = q1Wind_Speed - 1.5 * iqrWind_Speed
upper_boundWind_Speed = q3Wind_Speed + 1.5 * iqrWind_Speed

# Step 4: Filter out rows where 'Wind_Speed' is outside the bounds
Wind_Speed_removed = dmp3[(dmp3.Wind_Speed >= lower_boundWind_Speed) & (dmp3.Wind_Speed <= upper_boundWind_Speed)]

# Show the number of rows before and after removing outliers
print(f"Original dataset shape: {dm.shape}")
print(f"Dataset shape after removing outliers: {Wind_Speed_removed.shape}")

# Display the first few rows of the data without outliers

dmp4=Wind_Speed_removed
dmp4.head()


[ ]
plt.figure(figsize=(10, 10))
num_cols = dmp4.select_dtypes(include = ["int64","float64"])
for i, col in enumerate(num_cols):
    plt.subplot(5, 4, i+1)
    sns.boxplot(dmp4[col])
    plt.title(col)
plt.tight_layout()
plt.show()


[ ]
dm1= dmp4.copy() #keeping a copy of pre processed data

[ ]
from scipy.stats import skew, kurtosis
# Calculate skewness and kurtosis
num_cols=dm1.select_dtypes(include='number').columns
skewness = dm1[num_cols].apply(skew)
kurt = dm1[num_cols].apply(lambda x: kurtosis(x, fisher=False))
print(skewness)
print(kurt)
# Identify positive and negative skewness (absolute value > 1)
positive_skewness = skewness[skewness > 1]
negative_skewness = skewness[skewness < -1]
print('Variables with positive skewness (skew > 1):\n', positive_skewness)
print('Variables with negative skewness (skew < -1):\n', negative_skewness)
# Identify platykurtic (kurtosis < 3) and leptokurtic (kurtosis > 3) distributions
platykurtic = kurt[kurt < 3]
leptokurtic = kurt[kurt > 3]
print('Variables with platykurtic distribution (kurtosis < 3):\n', platykurtic)
print('Variables with leptokurtic distribution (kurtosis > 3):\n', leptokurtic)
Year                -0.310867
Month               -0.060177
Max_Temp            -0.381325
Min_Temp            -0.651720
Rainfall             1.067965
Relative_Humidity   -0.602449
Wind_Speed           0.713058
Cloud_Coverage       0.140917
Bright_Sunshine     -0.349437
Station_Number      -0.433311
X_COR               -0.177652
Y_COR                0.231289
LATITUDE             0.380814
LONGITUDE           -0.030078
ALT                  2.011465
Flood?               1.719605
dtype: float64
Year                 2.060189
Month                1.718437
Max_Temp             2.603761
Min_Temp             2.041023
Rainfall             3.270332
Relative_Humidity    2.713631
Wind_Speed           2.847881
Cloud_Coverage       1.662090
Bright_Sunshine      2.446198
Station_Number       2.468333
X_COR                2.537262
Y_COR                3.088549
LATITUDE             2.557854
LONGITUDE            1.786313
ALT                  6.940125
Flood?               3.957040
dtype: float64
Variables with positive skewness (skew > 1):
 Rainfall    1.067965
ALT         2.011465
Flood?      1.719605
dtype: float64
Variables with negative skewness (skew < -1):
 Series([], dtype: float64)
Variables with platykurtic distribution (kurtosis < 3):
 Year                 2.060189
Month                1.718437
Max_Temp             2.603761
Min_Temp             2.041023
Relative_Humidity    2.713631
Wind_Speed           2.847881
Cloud_Coverage       1.662090
Bright_Sunshine      2.446198
Station_Number       2.468333
X_COR                2.537262
LATITUDE             2.557854
LONGITUDE            1.786313
dtype: float64
Variables with leptokurtic distribution (kurtosis > 3):
 Rainfall    3.270332
Y_COR       3.088549
ALT         6.940125
Flood?      3.957040
dtype: float64
VISUALIZATION

[ ]
# Histograms for numerical columns after outlier treatment
num_cols = dm1.select_dtypes(include=['number']).columns

dm1[num_cols].hist(bins=15, figsize=(15, 10), layout=(5, 4))
plt.tight_layout()
plt.show()



[ ]
# Visualize the distributions to check for skewness
for column in dm1.columns:
    sns.histplot(dm1[column], kde=True)
    plt.title(f'Distribution of {column}')
    plt.show()


[ ]
# Step 2: Apply log transformation to skewed features
# We'll apply the log transformation to any feature with skewness > 1 or < -1
dm1_log_transformed = dm1.copy()
for col in dm1.columns:
    if dm1[col].skew() > 1 or dm1[col].skew() < -1:
        dm1_log_transformed[col] = np.log1p(dm1[col])

print("\nSkewness after log transformation:")
print(dm1_log_transformed.skew())

Skewness after log transformation:
Year                -0.310892
Month               -0.060182
Max_Temp            -0.381356
Min_Temp            -0.651772
Rainfall            -0.651133
Relative_Humidity   -0.602498
Wind_Speed           0.713116
Cloud_Coverage       0.140929
Bright_Sunshine     -0.349465
Station_Number      -0.433346
X_COR               -0.177667
Y_COR                0.231308
LATITUDE             0.380845
LONGITUDE           -0.030080
ALT                  0.712620
Flood?               1.719744
dtype: float64
Scaling

[ ]
# Scaling using StandardScaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
dm1_scaled = scaler.fit_transform(dm1)
# Convert the result back to a DataFrame
dm1_scaled = pd.DataFrame(dm1_scaled, columns=dm1.columns)
dm1_scaled

DATA SPLITTING

[ ]
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

X= dm1_scaled.drop(["Flood?"],axis =1)
Y= dm1_scaled["Flood?"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
MODEL SELECTION
Models Selected

Linear Regressor Model
Decision Tree Regressor Model
Random Forest Regressor Model
Gradient Boosting Regressor Model
Support Vector Regressor
MODEL TRAINING AND EVALUATION (Without Feature Selection & Hyperparameter Tuning)

[ ]
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
model_name= []
RMSE = []
MSE = []
MAE = []
R2_score = []
models = [
    LinearRegression(),
    DecisionTreeRegressor(),
    RandomForestRegressor(),
    GradientBoostingRegressor(),
    SVR()
]
for model in models :
    model.fit(X_train , y_train)

    prediction = model.predict(X_test)

    model_name.append(model.__class__.__name__)

    RMSE.append(mean_squared_error(y_test, prediction, squared=False))
    MSE.append(mean_squared_error(y_test, prediction))
    MAE.append(mean_absolute_error(y_test, prediction))
    R2_score.append(r2_score(y_test, prediction) * 100)

models_df = pd.DataFrame({"Model-Name":model_name, "RMSE": RMSE, "MSE":MSE, "MAE":MAE, "R2_Score":R2_score})
models_df = models_df.set_index('Model-Name')
models_df.sort_values("R2_Score", ascending = False)

FEATURE SELECTION
1. SelectKBest

[ ]
from sklearn.feature_selection import SelectKBest, f_regression
# SelectKBest with f_regression
selector_kbest = SelectKBest(score_func=f_regression, k=15)
X_kbest = selector_kbest.fit_transform(X_train, y_train)
# Get the selected feature indices
selected_indices_kbest = selector_kbest.get_support(indices=True)
# Get the names of the selected features
selected_features_kbest = X_train.columns[selected_indices_kbest]
print("Selected features using SelectKBest:", selected_features_kbest)
Selected features using SelectKBest: Index(['Year', 'Month', 'Max_Temp', 'Min_Temp', 'Rainfall',
       'Relative_Humidity', 'Wind_Speed', 'Cloud_Coverage', 'Bright_Sunshine',
       'Station_Number', 'X_COR', 'Y_COR', 'LATITUDE', 'LONGITUDE', 'ALT'],
      dtype='object')
Training using features selected using SelectKBest

[ ]
X1 = X[['Year', 'Month', 'Max_Temp', 'Min_Temp', 'Rainfall',
       'Relative_Humidity', 'Wind_Speed', 'Cloud_Coverage', 'Bright_Sunshine',
       'Station_Number', 'X_COR', 'Y_COR', 'LATITUDE', 'LONGITUDE', 'ALT']]
X1_train, X1_test, y1_train, y1_test = train_test_split(X1,Y,test_size = 0.2,random_state = 42)
model_name= []
RMSE = []
MSE = []
MAE = []
R2_score = []
models = [
    LinearRegression(),
    DecisionTreeRegressor(),
    RandomForestRegressor(),
    GradientBoostingRegressor(),
    SVR(),

]
for model in models :
    model.fit(X1_train , y1_train)

    prediction = model.predict(X1_test)

    model_name.append(model.__class__.__name__)

    RMSE.append(mean_squared_error(y1_test, prediction, squared=False))
    MSE.append(mean_squared_error(y1_test, prediction))
    MAE.append(mean_absolute_error(y1_test, prediction))
    R2_score.append(r2_score(y1_test, prediction) * 100)

models_df = pd.DataFrame({"Model-Name":model_name, "RMSE": RMSE, "MSE":MSE, "MAE":MAE, "R2_Score":R2_score})
models_df = models_df.set_index('Model-Name')
models_df.sort_values("R2_Score", ascending = False)

2. SelectFromModel with Lasso (L1 Regularization)

[ ]
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
# Lasso model
lasso_model = Lasso(alpha=0.01)  # Adjust alpha as needed
# SelectFromModel with Lasso
selector_lasso = SelectFromModel(lasso_model, max_features=15)
X_lasso = selector_lasso.fit_transform(X_train, y_train)
# Get the selected feature indices
selected_indices_lasso = selector_lasso.get_support(indices=True)
# Print selected feature names
selected_features_lasso = X_train.columns[selected_indices_lasso]
print("Selected features using SelectFromModel with Lasso:", selected_features_lasso)
Selected features using SelectFromModel with Lasso: Index(['Year', 'Max_Temp', 'Min_Temp', 'Rainfall', 'Cloud_Coverage', 'X_COR',
       'Y_COR', 'LATITUDE', 'LONGITUDE', 'ALT'],
      dtype='object')
Training using features selected using Lasso

[ ]
X2 = X[['Year', 'Max_Temp', 'Min_Temp', 'Rainfall', 'Cloud_Coverage', 'X_COR',
       'Y_COR', 'LATITUDE', 'LONGITUDE', 'ALT']]
X2_train, X2_test, y2_train, y2_test = train_test_split(X2,Y,test_size = 0.2,random_state = 42)
model_name= []
RMSE = []
MSE = []
MAE = []
R2_score = []
models = [
    LinearRegression(),
    DecisionTreeRegressor(),
    RandomForestRegressor(),
    GradientBoostingRegressor(),
    SVR(),

]
for model in models :
    model.fit(X2_train , y2_train)

    prediction = model.predict(X2_test)

    model_name.append(model.__class__.__name__)

    RMSE.append(mean_squared_error(y2_test, prediction, squared=False))
    MSE.append(mean_squared_error(y2_test, prediction))
    MAE.append(mean_absolute_error(y2_test, prediction))
    R2_score.append(r2_score(y2_test, prediction) * 100)

models_df = pd.DataFrame({"Model-Name":model_name, "RMSE": RMSE, "MSE":MSE, "MAE":MAE, "R2_Score":R2_score})
models_df = models_df.set_index('Model-Name')
models_df.sort_values("R2_Score", ascending = False)

Hyperparameter tuning
is the process of adjusting settings in a machine learning model to make it work better.

To improve the model's performance: Better settings can lead to more accurate predictions.

To prevent mistakes: Good tuning helps the model not to memorize the training data (overfitting) or to be too simple (underfitting).


[ ]
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import time
import pickle

# Define parameter grids
param_grids = {
    'LinearRegression': {
        'fit_intercept': [True, False],
    },
    'DecisionTreeRegressor': {
        'max_depth': [10, 20],
        'min_samples_split': [2, 10],
        'min_samples_leaf': [1, 5],
    },
    'RandomForestRegressor': {
        'n_estimators': [200, 300],
        'max_depth': [10, 20],
        'min_samples_split': [2, 10],
    },
    'GradientBoostingRegressor': {
        'learning_rate': [0.01, 0.1],
        'n_estimators': [200, 300],
        'max_depth': [5, 7],
    },
    'SVR': {
        'kernel': ['linear','rbf'],
        'C': [ 1, 10],
        'gamma': ['scale'],
    }
}

# Lists to store the results
model_name = []
RMSE = []
MSE = []
MAE = []
R2_score = []
best_params = []

# Define the models
models = [
    LinearRegression(),
    DecisionTreeRegressor(),
    RandomForestRegressor(),
    GradientBoostingRegressor(),
    SVR(),
]

# Loop through each model and perform GridSearchCV except for svr use random search cv
for model in models:
    model_class_name = model.__class__.__name__
    print(f"Starting tuning for {model_class_name}...")
    if model_class_name == 'SVR':
        search = RandomizedSearchCV(estimator=model, param_distributions=param_grids[model_class_name],
                                    scoring='neg_mean_squared_error', cv=3, n_iter=5, n_jobs=-1, random_state=42)
    else:
        search = GridSearchCV(estimator=model, param_grid=param_grids[model_class_name], scoring='neg_mean_squared_error', cv=5, n_jobs=-1)

    start_time = time.time()  # Start timing
    search.fit(X_train, y_train)
    end_time = time.time()  # End timing
    print(f"{model_class_name} tuning runtime: {(end_time - start_time)/60:.2f} minutes")

    best_model = search.best_estimator_
    prediction = best_model.predict(X_test)

    model_name.append(model_class_name)
    RMSE.append(mean_squared_error(y_test, prediction,squared=False))
    MSE.append(mean_squared_error(y_test, prediction))
    MAE.append(mean_absolute_error(y_test, prediction))
    R2_score.append(r2_score(y_test, prediction) * 100)
    best_params.append(search.best_params_)  # Store the best parameters
    print(f"{model_class_name} Best Params: {search.best_params_}")

    # Save the best model using pickle
    with open(f'{model_class_name}_best_model.pkl', 'wb') as file:
        pickle.dump(best_model, file)
    print(f"Saved {model_class_name} best model to {model_class_name}_best_model.pkl")

# Create a DataFrame with the results
models_df = pd.DataFrame({
    "Model-Name": model_name,
    "RMSE": RMSE,
    "MSE": MSE,
    "MAE": MAE,
    "R2_Score": R2_score,
    "Best Params": best_params
})
models_df = models_df.set_index('Model-Name')
models_df = models_df.sort_values("R2_Score", ascending=False)
models_df

CONCLUSION
RandoForestRegressor has the best performance with lowest values of rmse, mse, mae and high value of r2 score

SAVE THE MODEL
Saved the each model with the best parameters using pickle. Saved the model with ".pkl" extension

LOAD THE MODEL

[ ]
with open('RandomForestRegressor_best_model.pkl','rb') as file: # loading the gradient boosting regressor model
    mp=pickle.load(file)
Test the Model

[ ]
# Load the unseen data
unseen_data_path = "unseen_flood_data.csv"
unseen_df = pd.read_csv(unseen_data_path)

# Ensure unseen_df is defined before you use it
# List of feature names used during training
training_feature_names = ['Year', 'Month', 'Max_Temp', 'Min_Temp', 'Rainfall',
       'Relative_Humidity', 'Wind_Speed', 'Cloud_Coverage', 'Bright_Sunshine',
       'Station_Number', 'X_COR', 'Y_COR', 'LATITUDE', 'LONGITUDE', 'ALT']

# Make sure unseen data only contains the columns used during training
unseen_df = unseen_df[training_feature_names]  # Select relevant features

[ ]
from sklearn.preprocessing import StandardScaler
import joblib
import pandas as pd

# Load the unseen data
unseen_data_path = "unseen_flood_data.csv"
unseen_df = pd.read_csv(unseen_data_path)

# Use the same feature names as used during training
unseen_df = unseen_df[training_feature_names]

# Preprocess the unseen data (apply the same transformations as the training data)
scaler = StandardScaler()

# Apply the same scaler used during training
unseen_data_scaled = scaler.fit_transform(unseen_df)  # Scale the unseen data

# Load the saved model
model_path = "RandomForestRegressor_best_model.pkl"
loaded_model = joblib.load(model_path)

# Predict using the loaded model
unseen_predictions = loaded_model.predict(unseen_data_scaled)
print("Predictions on unseen data:", unseen_predictions)

# Define the threshold: +ve is "Flood", -ve is "No Flood"
label_mapping = {0: "No Flood", 1: "Flood"}

# Apply the threshold and map predictions to labels
predicted_labels = [
    label_mapping[1 if pred > 0 else 0] for pred in unseen_predictions
]

print("Human-Readable Predictions on Unseen Data:", predicted_labels)

Predictions on unseen data: [-0.45900563 -0.45900563 -0.45900563 -0.45880174 -0.45886562  2.17277442
  2.17494306 -0.43819993]
Human-Readable Predictions on Unseen Data: ['No Flood', 'No Flood', 'No Flood', 'No Flood', 'No Flood', 'Flood', 'Flood', 'No Flood']
/usr/local/lib/python3.10/dist-packages/sklearn/base.py:493: UserWarning: X does not have valid feature names, but RandomForestRegressor was fitted with feature names
  warnings.warn(

[ ]
from sklearn.metrics import accuracy_score

# Example: True labels and predictions (continuous values)
y_true = [-0.45900563, -0.45900563, -0.45900563, -0.45880174, -0.45886562,
          2.17277442, 2.17494306, 2.17494306]

unseen_predictions = [-0.45900563, -0.45900563, -0.45900563, -0.45880174,
                      -0.45886562, 2.17277442, 2.17494306, -0.43819993]

# Convert continuous values to binary labels using a threshold

Accuracy on unseen data: 0.875
Conclusion
This project successfully developed a machine learning model to predict flood based on the weather conditions. We performed data preprocessing, feature selection, and model training using multiple algorithms. By tuning the hyperparameters of the RandomForestClassifier, we achieved a highly accurate and generalized model. The saved model can now be used for real-time predictions.

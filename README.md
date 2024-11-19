
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

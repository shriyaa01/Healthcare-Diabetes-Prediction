# Healthcare-Diabetes-Prediction
Diabetes Outcome Prediction using Machine Learning: This repository contains code for predicting diabetes outcomes based on various health attributes.

1. **Importing Libraries:**
   The necessary libraries are imported, including `numpy` as `np`, `pandas` as `pd`, `matplotlib.pyplot` as `plt`, and `seaborn` as `sns`. These libraries are commonly used for data manipulation, visualization, and analysis tasks.

2. **Loading Data:**
   The dataset named "diabetes.csv" is loaded into a Pandas DataFrame using the `pd.read_csv()` function. The data contains information about various attributes related to diabetes.

3. **Exploratory Data Analysis (EDA):**
   Several EDA steps are performed on the DataFrame:
   - `.shape`: Display the dimensions of the DataFrame (number of rows and columns).
   - `.head()`: Display the first few rows of the DataFrame.
   - `.tail()`: Display the last few rows of the DataFrame.
   - `.describe()`: Display summary statistics of the numerical columns.
   - `.isnull().sum()`: Check for missing values in each column.
   - `.value_counts()`: Count the occurrences of each unique value in the "Outcome" column.
   - `.groupby().mean()`: Calculate the mean values of different attributes grouped by the "Outcome" column.
   - `.corr()`: Calculate the correlation matrix of all numerical attributes.
   - Visualization: A heatmap is plotted using Seaborn to visualize the correlation matrix.

4. **Data Preprocessing:**
   The dataset is split into the dependent variable `Y` (the "Outcome" column) and the independent variables `X` (all other columns). The independent variables are standardized using the `StandardScaler` from `sklearn.preprocessing`.

5. **Train-Test Split:**
   The data is split into training and testing sets using the `train_test_split()` function from `sklearn.model_selection`. The `stratify` parameter ensures that the class distribution is preserved in both sets.

6. **Model Building:**
   - **Support Vector Machine (SVM):** A linear SVM model is created using the `svm.SVC()` class from `sklearn.svm`. It's fitted to the training data using the `.fit()` method and predictions are made on the test data.
   - **Logistic Regression:** A logistic regression model is created using the `LogisticRegression()` class from `sklearn.linear_model`. It's fitted to the training data and used to make predictions.
   - **K Nearest Neighbors (KNN):** A KNN classifier is created using the `KNeighborsClassifier()` class from `sklearn.neighbors`. It's fitted to the training data and used to make predictions.

7. **Making Predictions:**
   For each model, predictions are made on the test data using the respective models.

8. **Individual Model Evaluation:**
   The accuracy of each model is evaluated using the `accuracy_score()` function from `sklearn.metrics`. Accuracy measures how often the model's predictions match the actual outcomes.

9. **Making Individual Predictions:**
   The code provides a sample input data point and demonstrates how to preprocess and predict using the trained models. The standardized input data is used to predict whether the person is diabetic or not.

10. **Comparing Model Performance:**
    The accuracy scores of all three models are printed to compare their performance.


This code provides a comprehensive overview of loading, preprocessing, modeling, and evaluating different machine learning models for diabetes prediction.

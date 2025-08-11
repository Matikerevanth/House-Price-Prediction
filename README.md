# House-Price-Prediction

## overview 
This project predict house price predictions using 13 different machine learning regression models and provides an interaction web interface built with Streamlit.

This project is divided into two main parts:
1. Backend(Model Training & Evalution): Training multiple machine learning models, evaluates them and save them as .pkl files for future predictions.
2. Frontend(Streamlit App): Lodas the training models, allows users to input property details and predicts the house price in real time
   
   ## Dataset
   * File: USA_Housing.csv
   * Target Variable: Price
   * Features:
       -> Avg.Area income
       -> Avg.Area House Age
       -> Avg. Area Number of Rooms\Bedrooms
       -> Area Population
   * Dropped Columns:
        -> Address(non-numeric)

## Backend - Model Training & Evalution
1. Importing Libraries
   The backend uses:
   * pandas -> Data loading & manipulation
   * scikit-learn -> Regression models, metrics and pipelines
   * LightGBM & XGBoost -> Advanced boosting models
   * pickle -> Save models for future use

 2. Data Preprocessing
   X = data.drop(['Price', 'Address'], axis=1)
   y = data['Price']
 * Removed target variable from features(attributes)
 * Dropped Address column

3. Train-Test-Split
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
* 80% -> training
* 20% -> testing

4. Model Selection
   Trains 13 Regression models:
    1.Linear Regression
    2. Huber Regressor (robust regression)
    3. Ridge Regression
    4. Lasso Regression
    5. ElasticNet Regression
    6. Polynomial Regression (degree=4)
    7. SGDRegressor
    8. MLPRegressor (ANN)
    9. Random Forest
    10. SVR (Support Vector Regression)
    11. LightGBM Regressor
    12. XGBoost Regressor
    13. KNN Regressor

5. Training & Saving Models
   For each models:
    * Fit on training set
    * Predict on test set
    * Calculations
       -> MAE --> Mean Absolute Error
       -> MSE --> Mean Squared Error
       -> R² → Coefficient of Determination
  6. Saving Results
      results_df = pd.DataFrame(results)
      results_df.to_csv('model_evaluation_results.csv', index=False)

## Frontend --Streamlit App

1. Loading Models
   for name in model_names:
    with open(f'{name}.pkl', 'rb') as f:
        models[name] = pickle.load(f)

2. User interface
  * Title: "🏠 House Price Prediction App"
  * Model Selection: Choose from the trained models via dropdown
  * Input Fields:
     * Avg. Area Income
     * Avg. Area House Age
     * Avg. Area Number of Rooms
     * Avg. Area Number of Bedrooms
     * Area Population

3. Prediction
   When the "Predict" button is clicked:
   * Creates a DataFrame with user inputs
   * Uses the selected model to predict price
   * Displays prediction in a currency format
      
## How To Run The Code
streamlit run app.py


   
     
     

import pandas as pd
import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib
import numpy as np
from supabase import create_client
from sklearn.preprocessing import StandardScaler
from aerthos_quant.api.supabase import fetch_table

import sys
import os

def preprocess_data_train():
    carbon_data = pd.DataFrame(fetch_table("carbon"))
    carbon_data.index = pd.to_datetime(carbon_data['Date'])
    carbon_data = carbon_data[["Price", "Open", "High", "Low", "Vol", "pct_change"]]
    carbon_data = carbon_data.dropna()
    carbon_data = carbon_data.sort_index(ascending=True)
    carbon_data.index = carbon_data.index.tz_localize('UTC')

    
    EU_rate = pd.DataFrame(fetch_table("obs"))
    EU_rate.index = pd.to_datetime(EU_rate['Month'])
    EU_rate = EU_rate["2018-01-01": pd.Timestamp.today()]
    EU_rate.index = EU_rate.index.tz_localize('UTC')
    

    dow_data = yf.download("^DJI", start="2018-01-01", end=pd.Timestamp.today())
    dow_data.index = dow_data.index.tz_localize('UTC')

    stoxx_data = yf.download("^STOXX", start="2018-01-01", end=pd.Timestamp.today())
    stoxx_data.index = stoxx_data.index.tz_localize('UTC')
    

    firms = ["CEZ.PR", "CNA.L", "EBK.DE", "EDNR.MI", "EDP.LS", "ENEL.MI", "ENGI.PA", "EOAN.DE", "FORTUM.HE", "IBE.MC", "PPC.AT", "RWE.DE", "SSE.L", "VER.VI"]
    firms_data = yf.download(firms, start="2018-01-01", end=pd.Timestamp.today())
    
    if firms_data.index.tz is None:
        firms_data.index = firms_data.index.tz_localize('UTC')
    else:
        firms_data.index = firms_data.index.tz_convert('UTC')
    firm_data = firms_data["Close"].dropna(axis=1, how="all")

    currencies = ["EURUSD=X", "DX-Y.NYB", 'NG=F', 'BZ=F']
    currencies_data = yf.download(currencies, start="2018-01-01", end=pd.Timestamp.today())
    if currencies_data.index.tz is None:
        currencies_data.index = currencies_data.index.tz_localize('UTC')
    else:
        currencies_data.index = currencies_data.index.tz_convert('UTC')
    currencies_data["EUR_index"] = currencies_data["Close"]["EURUSD=X"] * currencies_data["Close"]["DX-Y.NYB"]

    # Combine all the data into one dataframe
    data = pd.concat([carbon_data, dow_data["Close"].reindex(carbon_data.index), 
                      stoxx_data["Close"].reindex(carbon_data.index), 
                      EU_rate["OBS_Value"].reindex(carbon_data.index), 
                      currencies_data["Close"]['NG=F'].reindex(carbon_data.index), 
                      currencies_data["Close"]['BZ=F'].reindex(carbon_data.index), 
                      currencies_data["EUR_index"].reindex(carbon_data.index), firm_data.reindex(carbon_data.index)], axis=1)

    # Change the name of the columns
    data.columns = ["Carbon_Price", "Open", "High", "Low", "Vol.",  "Carbon_Pct_change", "Dow_Adj_Close", "Stoxx_Adj_Close", 
                    "EU_rate", "NatGas_Adj_Close", "Brent_Adj_Close", "EUR_index"] + firm_data.columns.tolist() 
    

    data = data.ffill()
    data = data.dropna()
    
   
    
    # Generate features
    columns_to_transform = ['Carbon_Price']
    time_windows = [30, 90, 180]

    for window in time_windows:
        data[f'Carbon_Price_{window}d_Mean'] = data[columns_to_transform].rolling(window=window).mean()
        data[f'Carbon_Price_{window}d_Std'] = data[columns_to_transform].rolling(window=window).std()
        data[f'Carbon_Price_{window}d_Max'] = data[columns_to_transform].rolling(window=window).max()
        data[f'Carbon_Price_{window}d_Min'] = data[columns_to_transform].rolling(window=window).min()
        data[f'Carbon_Price_{window}d_Return'] = data[columns_to_transform].pct_change(periods=window)
    
     
    scaler = StandardScaler()
    columns = data.drop(columns=["Carbon_Price"]).columns
    index = data.index
    
    train_data = data["2018-01-01": "2025-01-01"]
    predict_data = data["2025-01-01": ]
    scaled_data = scaler.fit_transform(train_data.drop(columns=["Carbon_Price"]))
    scaled_data_predict = scaler.transform(predict_data.drop(columns=["Carbon_Price"]))
    
    return train_data[["Carbon_Price"]], pd.DataFrame(scaled_data, columns=columns, index=train_data.index),predict_data[["Carbon_Price"]], pd.DataFrame(scaled_data_predict, columns=columns, index=predict_data.index)


def model_train(train_Y, train_X):
    y_temp = train_Y.shift(-1)  # Temporary target variable for dataset splitting

    # Split the dataset (X_train, X_test, y_train, y_test)
    X_train, X_test, _, _ = train_test_split(train_X, y_temp, test_size=0.2, random_state=42)

    # Initialize a DataFrame to store final prediction results
    prediction_results = pd.DataFrame(index=X_test.index)  # Dates as index

    # Loop to generate 10 models, each predicting prices for the next n days
    for n in range(1, 11):
        print(f"Training model to predict prices for the next {n} days")
        
        # 1. Calculate future n days' prices (target variable)
        train_Y[f'Carbon_Price_Next_{n}_Day'] = train_Y['Carbon_Price'].shift(-n)  # Future n days' prices as target variable
        
        # Remove rows with NaN values, as future n days may cause missing values
        X = train_Y.dropna()
        
        # Ensure `X_train` and `X_test` indices are consistent with `X`
        common_train_index = X_train.index.intersection(X.index)
        common_test_index = X_test.index.intersection(X.index)
        
        X_train_aligned = X_train.loc[common_train_index]
        X_test_aligned = X_test.loc[common_test_index]
        
        # Use the same training and testing sets, only changing the target variable
        y_train = X.loc[common_train_index, f'Carbon_Price_Next_{n}_Day']
        y_test = X.loc[common_test_index, f'Carbon_Price_Next_{n}_Day']

        # 2. Create and train the Random Forest model
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train_aligned, y_train)

        # 3. Make predictions
        y_pred = rf_model.predict(X_test_aligned)

        # 4. Save prediction results to DataFrame, column name as 'Model_n_Prediction'
        prediction_results[f'Day_{n}_Prediction'] = pd.Series(y_pred, index=X_test_aligned.index)
        prediction_results[f'Day_{n}_True'] = pd.Series(y_test, index=X_test_aligned.index)

        # 5. Evaluate model performance
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        

        model_filename = f'../aerthos_quant/models/model_{n}_day.pkl'
        joblib.dump(rf_model, model_filename)

        print(f"Mean Squared Error (MSE) for the next {n} days prediction: {mse}")
        print(f"R-squared (R2) for the next {n} days prediction: {r2}")


    
def predict_prices(test_X, prediction_file_path):
    # Create a DataFrame for prediction results, using all dates from X as the index
    prediction_results = pd.DataFrame(index=test_X.index)

    # Load the model and predict prices for the next 10 days
    for n in range(1, 11):
        try:
            # Load the model
            model_path = f'../aerthos_quant/models/model_{n}_day.pkl'
            model = joblib.load(model_path)
            
            # Predict for the entire dataset
            predictions = model.predict(test_X)
            
            # Add the prediction results to the DataFrame
            prediction_results[f'Day_{n}'] = predictions 
            
            print(f"Completed prediction for day {n}, number of samples predicted: {len(predictions)}")
            
        except Exception as e:
            print(f"Error loading or predicting the model for day {n}: {str(e)}")

    # Print the prediction results
    print("\nSample of prediction results (first 5 rows):")
    print(prediction_results)
    print("\nShape of prediction results:", prediction_results.shape)

    # Save prediction results to CSV file
    prediction_results.to_csv(prediction_file_path)

def predict_prices_from_file(file_path):
    prediction_file_path = file_path

    train_Y, train_X, predict_Y, predict_X = preprocess_data_train()
    model_train(train_Y, train_X)

    predict_prices(predict_X, prediction_file_path)


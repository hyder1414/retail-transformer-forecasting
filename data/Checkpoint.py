# train.py

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor  
from sklearn.linear_model import LinearRegression  
from Preprocess1 import prepare_walmart_data

def main():
    #Loading preprocessed sequences
    X, y, scaler, feature_cols = prepare_walmart_data(
        seq_len=16,
        horizon=4,
        data_dir="cache",
        force_recompute=False,
        base_dir=".")

    print("X shape:", X.shape) 
    print("y shape:", y.shape)   
    print("num features:", len(feature_cols))

    # predicting only the next week 
    #Flatten the sequences 
    y_next = y[:, 0]   # log(Weekly_Sales + 1) for t+1
    N, T, F = X.shape
    X_flat = X.reshape(N, T * F)
    print("\nFlattened X shape for regression:", X_flat.shape) 
    # Train tesy validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X_flat,
        y_next,
        test_size=0.2,
        random_state=42,
        shuffle=True,)

    print("\nTrain size:", X_train.shape[0])
    print("Val size:", X_val.shape[0])

    # 5.RFS regressio for test purpose
    model = RandomForestRegressor(n_estimators=100,max_depth=10,random_state=42,n_jobs=-1, )
    model.fit(X_train, y_train)
    print("Evaluation")
    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_val, y_pred)
    print(f"RMSE: {rmse:.4f}")
    print(f"R^2:   {r2:.4f}")
    #    y_val and y_pred are log1p(sales). This is how to convert back to original sales scale
    y_val_sales = np.expm1(y_val)
    y_pred_sales = np.expm1(y_pred)

    print("\nSAMPLE PREDICTIONS")
    for i in range(5):
        print(
            f"True: {y_val_sales[i]:10.2f},  Pred: {y_pred_sales[i]:10.2f}"
        )
if __name__ == "__main__":
    main()

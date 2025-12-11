# Save this as fix_inf.py
import numpy as np

print("Loading data...")
data = np.load('windows_L52_H4.npz')

X = data['X']
y = data['y']
dates = data['dates']
holidays = data['holidays']
stores = data['stores']
depts = data['depts']
feature_cols = data['feature_cols']

print(f"X shape: {X.shape}")
print(f"Before: X has Inf: {np.isinf(X).any()}")
print(f"Before: X max: {np.nanmax(X[~np.isinf(X)])}")

# Replace inf with 0
X = np.where(np.isinf(X), 0, X)

# Clip extreme values to reasonable range
X = np.clip(X, -1000, 1000)

print(f"After: X has Inf: {np.isinf(X).any()}")
print(f"After: X has NaN: {np.isnan(X).any()}")
print(f"After: X min: {np.min(X)}")
print(f"After: X max: {np.max(X)}")
print(f"After: X mean: {np.mean(X)}")

print("\nSaving cleaned data...")
np.savez_compressed(
    'windows_L52_H4.npz',
    X=X,
    y=y,
    dates=dates,
    holidays=holidays,
    stores=stores,
    depts=depts,
    feature_cols=feature_cols,
    input_length=np.array(52),
    output_length=np.array(4),
)

print("Done! Data cleaned.")
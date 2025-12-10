"""
Evaluation metrics for Walmart sales forecasting.
Includes WMAE (Weighted Mean Absolute Error) - the competition metric!
"""

import numpy as np
import torch
from typing import Optional


def calculate_wmae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    is_holiday: np.ndarray,
    weights: Optional[np.ndarray] = None
) -> float:
    """
    Calculate Weighted Mean Absolute Error (WMAE) - Walmart competition metric.
    
    THIS IS THE METRIC THAT MATTERS FOR THE COMPETITION!
    
    Args:
        y_true: True sales values in DOLLARS (not log-transformed)
        y_pred: Predicted sales values in DOLLARS (not log-transformed)
        is_holiday: Boolean array indicating holiday weeks
        weights: Optional custom weights. If None, uses 5 for holidays, 1 otherwise
    
    Returns:
        WMAE score (lower is better)
        
    Formula:
        WMAE = Σ(w_i * |y_true_i - y_pred_i|) / Σ(w_i)
        where w_i = 5 if holiday, 1 otherwise
        
    Benchmark scores:
        - 1st place (2014): ~2,300
        - Top 5%: < 2,500
        - Top 10%: < 2,800
        - Top 25%: < 3,200
    """
    if weights is None:
        weights = np.where(is_holiday, 5.0, 1.0)
    
    errors = np.abs(y_true - y_pred)
    weighted_errors = weights * errors
    
    wmae = np.sum(weighted_errors) / np.sum(weights)
    return float(wmae)


def calculate_wmae_torch(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    is_holiday: torch.Tensor,
    weights: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    PyTorch version of WMAE for use in loss functions.
    
    Can be used as a training loss to optimize directly for the competition metric!
    """
    if weights is None:
        weights = torch.where(is_holiday, 5.0, 1.0)
    
    errors = torch.abs(y_true - y_pred)
    weighted_errors = weights * errors
    
    wmae = torch.sum(weighted_errors) / torch.sum(weights)
    return wmae


def inverse_log_transform(y_log: np.ndarray) -> np.ndarray:
    """
    Convert log1p transformed predictions back to original scale (dollars).
    
    Args:
        y_log: Log-transformed values (using log1p)
    
    Returns:
        Original scale values (dollars)
    """
    return np.expm1(y_log)


def inverse_log_transform_torch(y_log: torch.Tensor) -> torch.Tensor:
    """PyTorch version of inverse log transform"""
    return torch.expm1(y_log)


def calculate_metrics(
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
    is_holiday: np.ndarray
) -> dict:
    """
    Calculate all evaluation metrics.
    
    Args:
        y_true_log: True values in log space (log1p)
        y_pred_log: Predicted values in log space (log1p)
        is_holiday: Holiday flags
    
    Returns:
        Dictionary with:
        - mse_log: MSE in log space (training metric)
        - rmse: RMSE in dollar space
        - mae: MAE in dollar space  
        - wmae: Weighted MAE (COMPETITION METRIC!)
    """
    # Convert to dollar space
    y_true_dollars = inverse_log_transform(y_true_log)
    y_pred_dollars = inverse_log_transform(y_pred_log)
    
    # Clip to non-negative (sales can't be negative)
    y_pred_dollars = np.maximum(y_pred_dollars, 0)
    
    # Calculate metrics
    mse_log = np.mean((y_true_log - y_pred_log) ** 2)
    rmse = np.sqrt(np.mean((y_true_dollars - y_pred_dollars) ** 2))
    mae = np.mean(np.abs(y_true_dollars - y_pred_dollars))
    wmae = calculate_wmae(y_true_dollars, y_pred_dollars, is_holiday)
    
    return {
        'mse_log': float(mse_log),
        'rmse': float(rmse),
        'mae': float(mae),
        'wmae': float(wmae),  # ⭐ THIS IS THE COMPETITION METRIC!
    }


def evaluate_model(
    model,
    dataloader,
    device: str = 'cpu'
) -> dict:
    """
    Evaluate model on a dataset and calculate all metrics including WMAE.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader (must have dataset with get_all_metadata method)
        device: Device to run on
    
    Returns:
        Dictionary of metrics including WMAE
    """
    model.eval()
    
    all_y_true = []
    all_y_pred = []
    
    with torch.no_grad():
        for X, y_true in dataloader:
            X = X.to(device)
            y_pred = model(X)
            
            all_y_true.append(y_true.cpu().numpy())
            all_y_pred.append(y_pred.cpu().numpy())
    
    # Concatenate all batches
    y_true_all = np.concatenate(all_y_true, axis=0).flatten()
    y_pred_all = np.concatenate(all_y_pred, axis=0).flatten()
    
    # Get holiday flags from dataset
    metadata = dataloader.dataset.get_all_metadata()
    holidays_all = metadata['holidays'].flatten()
    
    # Calculate metrics
    metrics = calculate_metrics(y_true_all, y_pred_all, holidays_all)
    
    return metrics


def print_metrics(metrics: dict, prefix: str = ""):
    """Pretty print metrics with competition context"""
    print(f"{prefix}Metrics:")
    print(f"  MSE (log):  {metrics['mse_log']:.4f}")
    print(f"  RMSE ($):   ${metrics['rmse']:,.2f}")
    print(f"  MAE ($):    ${metrics['mae']:,.2f}")
    print(f"  WMAE ($):   ${metrics['wmae']:,.2f}  ⭐ COMPETITION METRIC")
    
    # Add context
    wmae = metrics['wmae']
    if wmae < 2300:
        print(f"               🏆 BETTER THAN 1ST PLACE (2014)!")
    elif wmae < 2500:
        print(f"               🥇 Top 5% performance!")
    elif wmae < 2800:
        print(f"               🥈 Top 10% performance!")
    elif wmae < 3200:
        print(f"               🥉 Top 25% performance!")
    else:
        print(f"               📊 Baseline performance")


# Example and testing
if __name__ == "__main__":
    print("=" * 70)
    print("WMAE METRIC DEMONSTRATION")
    print("=" * 70)
    
    # Example data (in dollars)
    y_true = np.array([10000, 50000, 12000, 48000, 15000])
    y_pred = np.array([10500, 45000, 11500, 49000, 14500])
    is_holiday = np.array([False, True, False, True, False])
    
    print("\nExample data:")
    print("  True sales: ", y_true)
    print("  Predictions:", y_pred)
    print("  Holidays:   ", is_holiday)
    
    errors = np.abs(y_true - y_pred)
    print(f"\nAbsolute errors: {errors}")
    
    # Regular MAE (no weighting)
    mae = np.mean(errors)
    print(f"\nRegular MAE: ${mae:.2f}")
    
    # WMAE (5x weight for holidays)
    wmae = calculate_wmae(y_true, y_pred, is_holiday)
    print(f"WMAE (competition metric): ${wmae:.2f}")
    
    print("\n" + "=" * 70)
    print("NOTICE: WMAE is higher because holiday errors are weighted 5x!")
    print("=" * 70)
    
    # Show breakdown
    weights = np.where(is_holiday, 5, 1)
    weighted_errors = weights * errors
    print(f"\nWeights:         {weights}")
    print(f"Weighted errors: {weighted_errors}")
    print(f"Sum of weighted errors: {np.sum(weighted_errors)}")
    print(f"Sum of weights:         {np.sum(weights)}")
    print(f"WMAE = {np.sum(weighted_errors)} / {np.sum(weights)} = {wmae:.2f}")
    
    print("\n✅ This is why getting holiday predictions right is CRITICAL!")

import numpy as np
import pandas as pd
import patsy

from sklearn.linear_model import Lasso, Ridge, ElasticNet, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ---------------------------------------------------------------------------
# 1. Load data and create lagged variables
# ---------------------------------------------------------------------------
data = pd.read_csv("HW1/vixlarge.csv")
data.sort_values("DATE", inplace=True)
data.reset_index(drop=True, inplace=True)

data['VIX_lag'] = data['VIX'].shift(1)
data.dropna(inplace=True)

# ---------------------------------------------------------------------------
# 2. Choose forecast horizons
# ---------------------------------------------------------------------------
horizons = [1, 5, 10, 22]

# ---------------------------------------------------------------------------
# 3. Rolling window parameters & model tuning
# ---------------------------------------------------------------------------
window_length = 3000

lasso_alpha = 0.5
ridge_alpha = 1.0
elasticnet_alpha = 0.5
elasticnet_l1_ratio = 0.5  # Mix of Lasso (L1) and Ridge (L2)
tree_depth = 5

# For the spline:
spline_formula = "cr(VIX_lag, df=5) - 1"

results_list = []

# ---------------------------------------------------------------------------
# 4. Main loop over each forecast horizon h
# ---------------------------------------------------------------------------
for h in horizons:
    # 4a. Create the "lead-h" target
    lead_col = f'VIX_lead_{h}'
    data[lead_col] = data['VIX'].shift(-h)
    
    # Drop final h rows (NaN in the target)
    tmp = data.dropna(subset=[lead_col]).copy()
    
    # 4b. Create your design matrices for (lag-based) models:
    X_lin = np.column_stack([
        np.ones(len(tmp)),      # intercept
        tmp['VIX_lag'].values   # single predictor
    ])
    y_all = tmp[lead_col].values
    
    # Create a design matrix for the natural spline
    X_spline_all = patsy.dmatrix(spline_formula, data=tmp, return_type='dataframe')
    
    n = len(y_all)
    
    # 4c. Storage for rolling predictions
    preds_lasso  = []
    preds_ridge  = []
    preds_elasticnet = []
    preds_tree   = []
    preds_spline = []
    actual_vals  = []
    
    # 4d. Rolling window loop
    for start_idx in range(n - window_length):
        end_idx = start_idx + window_length
        
        # Training sets
        X_train_lin = X_lin[start_idx : end_idx, :]
        X_train_spl = X_spline_all.iloc[start_idx : end_idx, :]
        y_train     = y_all[start_idx : end_idx]
        
        # The "test" point in the rolling scheme
        X_test_lin = X_lin[end_idx, :].reshape(1, -1)
        X_test_spl = X_spline_all.iloc[end_idx, :].values.reshape(1, -1)
        y_test     = y_all[end_idx]
        
        # --- Lasso ---
        lasso_model = Lasso(alpha=lasso_alpha)
        lasso_model.fit(X_train_lin, y_train)
        preds_lasso.append(lasso_model.predict(X_test_lin)[0])
        
        # --- Ridge ---
        ridge_model = Ridge(alpha=ridge_alpha)
        ridge_model.fit(X_train_lin, y_train)
        preds_ridge.append(ridge_model.predict(X_test_lin)[0])
        
        # --- ElasticNet ---
        elasticnet_model = ElasticNet(alpha=elasticnet_alpha, l1_ratio=elasticnet_l1_ratio)
        elasticnet_model.fit(X_train_lin, y_train)
        preds_elasticnet.append(elasticnet_model.predict(X_test_lin)[0])
        
        # --- Regression Tree ---
        tree_model = DecisionTreeRegressor(max_depth=tree_depth)
        tree_model.fit(X_train_lin, y_train)
        preds_tree.append(tree_model.predict(X_test_lin)[0])
        
        # --- Natural Cubic Spline + LinearRegression ---
        spline_model = LinearRegression()
        spline_model.fit(X_train_spl, y_train)
        preds_spline.append(spline_model.predict(X_test_spl)[0])
        
        # Store actual
        actual_vals.append(y_test)
    
    # Convert lists to arrays
    preds_lasso  = np.array(preds_lasso)
    preds_ridge  = np.array(preds_ridge)
    preds_elasticnet = np.array(preds_elasticnet)
    preds_tree   = np.array(preds_tree)
    preds_spline = np.array(preds_spline)
    actual_vals  = np.array(actual_vals)
    
    # 4e. Compute metrics
    mse_lasso       = mean_squared_error(actual_vals, preds_lasso)
    mae_lasso       = mean_absolute_error(actual_vals, preds_lasso)
    
    mse_ridge       = mean_squared_error(actual_vals, preds_ridge)
    mae_ridge       = mean_absolute_error(actual_vals, preds_ridge)
    
    mse_elasticnet  = mean_squared_error(actual_vals, preds_elasticnet)
    mae_elasticnet  = mean_absolute_error(actual_vals, preds_elasticnet)
    
    mse_tree        = mean_squared_error(actual_vals, preds_tree)
    mae_tree        = mean_absolute_error(actual_vals, preds_tree)
    
    mse_spline      = mean_squared_error(actual_vals, preds_spline)
    mae_spline      = mean_absolute_error(actual_vals, preds_spline)
    
    # 4f. Store results
    results_list.append({
        'Horizon': h, 'Model': 'Lasso',
        'MSE': mse_lasso, 'MAE': mae_lasso,
        'Tuning': f'alpha={lasso_alpha}'
    })
    results_list.append({
        'Horizon': h, 'Model': 'Ridge',
        'MSE': mse_ridge, 'MAE': mae_ridge,
        'Tuning': f'alpha={ridge_alpha}'
    })
    results_list.append({
        'Horizon': h, 'Model': 'ElasticNet',
        'MSE': mse_elasticnet, 'MAE': mae_elasticnet,
        'Tuning': f'alpha={elasticnet_alpha}, l1_ratio={elasticnet_l1_ratio}'
    })
    results_list.append({
        'Horizon': h, 'Model': 'Tree',
        'MSE': mse_tree, 'MAE': mae_tree,
        'Tuning': f'max_depth={tree_depth}'
    })
    results_list.append({
        'Horizon': h, 'Model': 'NaturalSpline',
        'MSE': mse_spline, 'MAE': mae_spline,
        'Tuning': f'df=5'
    })

# ---------------------------------------------------------------------------
# 5. Summarize in a DataFrame
# ---------------------------------------------------------------------------
results_df = pd.DataFrame(results_list)
results_df = results_df[['Horizon','Model','MSE','MAE','Tuning']]
print(results_df)
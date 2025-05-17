import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import TimeSeriesSplit
from category_encoders import TargetEncoder
import warnings
import lightgbm as lgbm
warnings.filterwarnings('ignore')

# Configuration
TARGET_COL = 'Price'
SEED = 42
TIME_COL = 'Scraped_Time'

# Load and sort data
train = pd.read_csv('train.csv').sort_values(TIME_COL)
test = pd.read_csv('test.csv').sort_values(TIME_COL)

# Feature Engineering Functions
def create_features(df):
    df = df.copy()
    
    # Convert Year to numeric, handling any non-numeric values
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    
    # Convert Kilometers to numeric, handling any non-numeric values
    df['Kilometers'] = pd.to_numeric(df['Kilometers'], errors='coerce')
    
    # Temporal features
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df['Year_Sold'] = df[TIME_COL].dt.year
    df['Month_Sold'] = df[TIME_COL].dt.month
    df['Car_Age'] = (df['Year_Sold'] - df['Year']).fillna(0).astype(int)
    
    # Interaction features
    df['Brand_Model'] = df['Car Make'] + '_' + df['Model']
    df['Fuel_Engine'] = df['Fuel'] + '_' + df['Engine Size (cc)'].astype(str)
    
    # Binning
    # Handle zero values separately for Kilometers binning
    non_zero_mask = df['Kilometers'] > 0
    if non_zero_mask.any():
        non_zero_km = df.loc[non_zero_mask, 'Kilometers']
        km_quantiles = non_zero_km.quantile([0.2, 0.4, 0.6, 0.8])
        km_bins = [0] + km_quantiles.tolist() + [float('inf')]
        df['Mileage_Bin'] = pd.cut(df['Kilometers'], bins=km_bins, labels=False, include_lowest=True)
    else:
        df['Mileage_Bin'] = 0
    
    df['Age_Bin'] = pd.cut(df['Car_Age'], bins=[0, 2, 5, 10, 20, 100], labels=False)
    
    # EV Features
    df['EV_Flag'] = df['Fuel'].isin(['Electric', 'Hybrid']).astype(int)
    df['Battery_Capacity'] = df['Battery Capacity'] * df['EV_Flag']
    
    # Option counts
    for col in ['Interior Options', 'Exterior Options', 'Technology Options']:
        df[col+'_count'] = df[col].fillna('').str.count(',') + 1
        
    return df.drop(columns=[TIME_COL])

# Preprocessing
def preprocess(df):
    # Handle missing values
    if 'Description_Score' in df.columns:
        df['Description_Score'] = df['Description_Score'].fillna(50)
    elif 'Description Score' in df.columns:
        df['Description_Score'] = df['Description Score'].fillna(50)
    else:
        df['Description_Score'] = 50  # Default value if column doesn't exist
    
    # Handle Paint column
    if 'Paint' in df.columns:
        df['Paint'] = df['Paint'].fillna('Unknown')
    
    # Convert binary features
    binary_map = {'Yes': 1, 'No': 0, 'New': 1, 'Used': 0}
    for col in ['Insurance', 'Condition', 'Car Customs']:
        if col in df.columns:
            df[col] = df[col].map(binary_map).fillna(0)
        
    return df

# Outlier handling
def handle_outliers(df):
    # Convert numeric columns to float type
    numeric_cols = ['Kilometers', 'Engine Size (cc)', 'Price']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # Calculate cap only if we have valid numeric values
            if df[col].notna().any():
                cap = df[col].quantile(0.995)
                df[col] = np.where(df[col] > cap, cap, df[col])
    return df

# Pipeline
feature_pipeline = Pipeline([
    ('create_features', FunctionTransformer(create_features)),
    ('preprocess', FunctionTransformer(preprocess)),
    ('outliers', FunctionTransformer(handle_outliers)),
])

# Transform data
X_train = feature_pipeline.fit_transform(train)
y_train = np.log1p(X_train[TARGET_COL])  # Log-transform target
X_test = feature_pipeline.transform(test)

# Remove target from features
X_train = X_train.drop(columns=[TARGET_COL])

# Define categorical features
cat_features = ['Car Make', 'Model', 'City', 'Fuel', 'Transmission',
                'Brand_Model', 'Fuel_Engine', 'Regional Specs', 'Paint',
                'Body Type', 'Exterior Color', 'Interior Color', 'Payment Method',
                'Neighborhood', 'Category', 'Subcategory']

# Create separate copies for different models
X_train_catboost = X_train.copy()
X_test_catboost = test.copy()
X_train_lgbm = X_train.copy()
X_test_lgbm = test.copy()

# Ensure test data has all the same columns as training data
missing_cols = set(X_train.columns) - set(X_test_lgbm.columns)
for col in missing_cols:
    X_test_lgbm[col] = 0  # Fill missing columns with 0
    X_test_catboost[col] = 0

# Ensure columns are in the same order
X_test_lgbm = X_test_lgbm[X_train.columns]
X_test_catboost = X_test_catboost[X_train.columns]

# Target Encoding for LightGBM
encoder = TargetEncoder(cols=cat_features, smoothing=20)
X_train_lgbm_encoded = encoder.fit_transform(X_train_lgbm, y_train)
X_test_lgbm_encoded = encoder.transform(X_test_lgbm)

# For CatBoost, convert numeric features to float and categorical features to string
# First, identify truly numeric columns by attempting conversion
numeric_features = []
for col in X_train_catboost.columns:
    if col not in cat_features:  # Skip known categorical features
        try:
            pd.to_numeric(X_train_catboost[col], errors='raise')
            numeric_features.append(col)
        except:
            cat_features.append(col)  # Add to categorical features if conversion fails

# Convert numeric features to float
for col in numeric_features:
    X_train_catboost[col] = pd.to_numeric(X_train_catboost[col], errors='coerce').fillna(0)
    X_test_catboost[col] = pd.to_numeric(X_test_catboost[col], errors='coerce').fillna(0)

# Convert categorical features to string
for col in cat_features:
    if col in X_train_catboost.columns:
        X_train_catboost[col] = X_train_catboost[col].astype(str)
        X_test_catboost[col] = X_test_catboost[col].astype(str)

# Get categorical feature indices for CatBoost
cat_indices = [X_train_catboost.columns.get_loc(col) for col in cat_features if col in X_train_catboost.columns]

# Model Ensemble
cat_model = CatBoostRegressor(
    iterations=1500,
    learning_rate=0.08,
    depth=7,
    l2_leaf_reg=4,
    random_seed=SEED,
    cat_features=cat_indices,
    silent=True
)

lgbm_model = LGBMRegressor(
    num_leaves=31,
    learning_rate=0.05,
    n_estimators=1200,
    reg_alpha=0.1,
    reg_lambda=0.1,
    verbose=1,
    random_state=SEED
)

# Time Series Cross-Validation
tscv = TimeSeriesSplit(n_splits=5)
scores = []
test_preds = []

for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
    print(f"\nFold {fold+1}")
    
    # Split data for both models
    X_tr_cat, X_val_cat = X_train_catboost.iloc[train_idx], X_train_catboost.iloc[val_idx]
    X_tr_lgbm, X_val_lgbm = X_train_lgbm_encoded.iloc[train_idx], X_train_lgbm_encoded.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    # Train CatBoost
    cat_model.fit(
        X_tr_cat, y_tr,
        eval_set=(X_val_cat, y_val),
        early_stopping_rounds=100,
        verbose=100
    )
    
    # Train LightGBM
    lgbm_model.fit(
        X_tr_lgbm, y_tr,
        eval_set=[(X_val_lgbm, y_val)],
        callbacks=[lgbm.early_stopping(stopping_rounds=100)],
        
    )
    
    # Ensemble predictions
    cat_pred = cat_model.predict(X_val_cat)
    lgbm_pred = lgbm_model.predict(X_val_lgbm)
    blend_pred = 0.7*cat_pred + 0.3*lgbm_pred
    
    # Calculate score
    fold_score = np.sqrt(np.mean((np.expm1(blend_pred) - np.expm1(y_val))**2))
    scores.append(fold_score)
    print(f"Fold {fold+1} RMSE: {fold_score:.2f}")
    
    # Generate test predictions
    test_preds.append(
        0.7*cat_model.predict(X_test_catboost) + 
        0.3*lgbm_model.predict(X_test_lgbm_encoded)
    )

# Final Prediction
final_pred = np.expm1(np.mean(test_preds, axis=0))

# Create submission
submission = pd.DataFrame({
    'Id': test['Id'],
    'Price': final_pred
})
submission.to_csv('submission2.csv', index=False)
print(f"Average CV RMSE: {np.mean(scores):.2f}")
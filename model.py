import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import TimeSeriesSplit
from category_encoders import TargetEncoder
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Load data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Sort data by Scraped_Time
train = train.sort_values('Scraped_Time')
test = test.sort_values('Scraped_Time')

# Define preprocessing functions
def preprocess_data(df, train_medians=None, is_train=True):
    # Create a copy to avoid modifying the original data
    df = df.copy()
    
    # Convert Year to numeric
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    # Convert Kilometers to numeric
    df['Kilometers'] = pd.to_numeric(df['Kilometers'], errors='coerce')
    
    # Convert Scraped_Time to datetime and extract year
    df['Scraped_Time'] = pd.to_datetime(df['Scraped_Time'])
    df['Scraped_Year'] = df['Scraped_Time'].dt.year
    df['Age'] = df['Scraped_Year'] - df['Year']
    df.drop(['Scraped_Time', 'Scraped_Year'], axis=1, inplace=True)
    
    # Annual Kilometers
    df['Annual_Kilometers'] = df['Kilometers'] / (df['Age'] + 1)
    
    # Handle Battery features for non-EV/Hybrid
    df['Battery Capacity'] = np.where(df['Fuel'].isin(['Electric', 'Hybrid']), df['Battery Capacity'], 0)
    df['Battery Range'] = np.where(df['Fuel'].isin(['Electric', 'Hybrid']), df['Battery Range'], 0)
    df['Battery Capacity'] = df['Battery Capacity'].fillna(0)
    df['Battery Range'] = df['Battery Range'].fillna(0)
    
    # Count options for Interior, Exterior, Technology
    option_cols = ['Interior Options', 'Exterior Options', 'Technology Options']
    for col in option_cols:
        df[col] = df[col].fillna('')
        df[col + '_count'] = df[col].apply(lambda x: 0 if x == '' else len(x.split(',')))
    df.drop(option_cols, axis=1, inplace=True)
    
    # Ordinal encoding
    body_condition_mapping = {'Excellent': 3, 'Good': 2, 'Fair': 1, 'Damaged': 0}
    paint_mapping = {'Original': 1, 'Repainted': 0}
    car_license_mapping = {'Valid': 2, 'Expired': 1, 'Not Applicable': 0, 'Missing': 0}
    car_customs_mapping = {'Cleared': 1, 'Not Cleared': 0, 'Missing': 0}
    condition_mapping = {'New': 1, 'Used': 0}
    insurance_mapping = {'Yes': 1, 'No': 0, 'Missing': 0}
    
    df['Body Condition'] = df['Body Condition'].map(body_condition_mapping).fillna(0).astype(int)
    df['Paint'] = df['Paint'].map(paint_mapping).fillna(0).astype(int)
    df['Car License'] = df['Car License'].map(car_license_mapping).fillna(0).astype(int)
    df['Car Customs'] = df['Car Customs'].map(car_customs_mapping).fillna(0).astype(int)
    df['Condition'] = df['Condition'].map(condition_mapping).fillna(0).astype(int)
    df['Insurance'] = df['Insurance'].map(insurance_mapping).fillna(0).astype(int)
    
    # Fill missing categorical values with 'Missing'
    categorical_cols = ['Car Make', 'Model', 'Trim', 'Body Type', 'Fuel', 'Transmission',
                        'Exterior Color', 'Interior Color', 'Regional Specs', 'Payment Method', 'City',
                        'Neighborhood', 'Category', 'Subcategory']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna('Missing')
            df[col] = df[col].astype(str)  # Ensure categorical columns are strings
    
    # Fill missing numerical values
    numerical_cols = ['Year', 'Kilometers', 'Number of Seats', 'Engine Size (cc)',
                      'Description_Score', 'Battery Capacity', 'Battery Range', 'Age',
                      'Annual_Kilometers']
    
    if is_train:
        # For training data, compute medians
        for col in numerical_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                median = df[col].median()
                if np.isnan(median):
                    median = 0
                df[col] = df[col].fillna(median)
        
        # Add interaction features
        df['Brand_Model'] = df['Car Make'] + '_' + df['Model']
        
        # Bin Year and Kilometers
        bins = [1990, 2000, 2010, 2020, 2024]
        df['Year_Bin'] = pd.cut(df['Year'], bins=bins, labels=False)
        
        # Create Kilometers bins using custom quantiles to handle zeros
        non_zero_mask = df['Kilometers'] > 0
        if non_zero_mask.any():
            non_zero_km = df.loc[non_zero_mask, 'Kilometers']
            km_quantiles = non_zero_km.quantile([0.2, 0.4, 0.6, 0.8])
            km_bins = [0] + km_quantiles.tolist() + [float('inf')]
            df['Kilometers_Bin'] = pd.cut(df['Kilometers'], bins=km_bins, labels=False, include_lowest=True)
        else:
            df['Kilometers_Bin'] = 0
        
        # Clip numerical features at 99th percentile
        for col in ['Kilometers', 'Engine Size (cc)', 'Annual_Kilometers']:
            cap = df[col].quantile(0.99)
            df[col] = df[col].clip(upper=cap)
        
        return df, df[numerical_cols].median()
    else:
        # For test data, use training medians
        if train_medians is not None:
            for col in numerical_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    df[col] = df[col].fillna(train_medians[col])
        
        # Add interaction features
        df['Brand_Model'] = df['Car Make'] + '_' + df['Model']
        
        # Bin Year and Kilometers
        bins = [1990, 2000, 2010, 2020, 2024]
        df['Year_Bin'] = pd.cut(df['Year'], bins=bins, labels=False)
        
        # Create Kilometers bins using custom quantiles to handle zeros
        non_zero_mask = df['Kilometers'] > 0
        if non_zero_mask.any():
            non_zero_km = df.loc[non_zero_mask, 'Kilometers']
            km_quantiles = non_zero_km.quantile([0.2, 0.4, 0.6, 0.8])
            km_bins = [0] + km_quantiles.tolist() + [float('inf')]
            df['Kilometers_Bin'] = pd.cut(df['Kilometers'], bins=km_bins, labels=False, include_lowest=True)
        else:
            df['Kilometers_Bin'] = 0
        
        return df

# Preprocess train and test data
train_preprocessed, train_medians = preprocess_data(train, is_train=True)
test_preprocessed = preprocess_data(test, train_medians=train_medians, is_train=False)

# Clip Price at 99th percentile
price_cap = train['Price'].quantile(0.99)
train['Price'] = train['Price'].clip(upper=price_cap)

# Separate features and target
X = train_preprocessed.drop('Price', axis=1)
y = np.log1p(train['Price'])  # Log transformation

# Initialize TimeSeriesSplit for cross-validation
tscv = TimeSeriesSplit(n_splits=5)

# Define target encoding columns
target_encode_cols = ['Model', 'City', 'Car Make']

# Create preprocessing pipeline
preprocessor = ColumnTransformer([
    ('target_enc', TargetEncoder(cols=target_encode_cols), target_encode_cols),
    ('num', 'passthrough', X.select_dtypes(include=['int64', 'float64']).columns)
])

# Initialize models
cat_features = [X.columns.get_loc(col) for col in X.select_dtypes(include=['object', 'category']).columns]
catboost_model = CatBoostRegressor(
    cat_features=cat_features,
    learning_rate=0.03,
    iterations=1500,
    l2_leaf_reg=8,
    depth=6,
    early_stopping_rounds=150,
    verbose=100,
    random_seed=42,
    eval_metric='RMSE',
    loss_function='RMSE'
)

lgbm_model = LGBMRegressor(
    num_leaves=32,
    max_depth=5,
    learning_rate=0.03,
    n_estimators=1500,
    random_state=42
)

# Train models with cross-validation
cv_scores_catboost = []
cv_scores_lgbm = []

# Create a dictionary to store category mappings
category_mappings = {}

for train_idx, val_idx in tscv.split(X):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # Create copies for different models
    X_train_catboost = X_train.copy()
    X_val_catboost = X_val.copy()
    X_train_lgbm = X_train.copy()
    X_val_lgbm = X_val.copy()
    
    # Apply target encoding
    encoder = TargetEncoder(cols=target_encode_cols)
    X_train_encoded = encoder.fit_transform(X_train, y_train)
    X_val_encoded = encoder.transform(X_val)
    
    # Process data for CatBoost
    for col in target_encode_cols:
        X_train_catboost[col] = X_train_encoded[col].round(3).astype(str)
        X_val_catboost[col] = X_val_encoded[col].round(3).astype(str)
    
    # Process data for LightGBM
    for col in X_train_lgbm.select_dtypes(include=['object', 'category']).columns:
        if col not in category_mappings:
            # Get unique values from both train and validation sets
            unique_values = pd.concat([X_train_lgbm[col], X_val_lgbm[col]]).unique()
            # Create mapping dictionary with a default value for unseen categories
            category_mappings[col] = {val: idx for idx, val in enumerate(unique_values)}
            category_mappings[col]['_unknown'] = len(unique_values)
        
        # Transform using the mapping, with fallback to unknown category
        X_train_lgbm[col] = X_train_lgbm[col].map(lambda x: category_mappings[col].get(x, category_mappings[col]['_unknown']))
        X_val_lgbm[col] = X_val_lgbm[col].map(lambda x: category_mappings[col].get(x, category_mappings[col]['_unknown']))
    
    # Train CatBoost
    catboost_model.fit(
        X_train_catboost, y_train,
        eval_set=(X_val_catboost, y_val),
        use_best_model=True,
        verbose=100
    )
    cv_scores_catboost.append(catboost_model.best_score_['validation']['RMSE'])
    
    # Train LightGBM
    lgbm_model.fit(X_train_lgbm, y_train)
    lgbm_pred = lgbm_model.predict(X_val_lgbm)
    cv_scores_lgbm.append(np.sqrt(np.mean((y_val - lgbm_pred) ** 2)))

print(f'CatBoost CV RMSE scores: {cv_scores_catboost}')
print(f'Mean CatBoost CV RMSE: {np.mean(cv_scores_catboost):.4f}')
print(f'LightGBM CV RMSE scores: {cv_scores_lgbm}')
print(f'Mean LightGBM CV RMSE: {np.mean(cv_scores_lgbm):.4f}')

# Train final models on full training data
encoder = TargetEncoder(cols=target_encode_cols)
X_encoded = encoder.fit_transform(X, y)

# Create copies for different models
X_catboost = X.copy()
X_lgbm = X.copy()
test_catboost = test_preprocessed.copy()
test_lgbm = test_preprocessed.copy()

# Process data for CatBoost
for col in target_encode_cols:
    X_catboost[col] = X_encoded[col].round(3).astype(str)
    test_catboost[col] = X_encoded[col].round(3).astype(str)

# Process data for LightGBM
for col in X_lgbm.select_dtypes(include=['object', 'category']).columns:
    if col not in category_mappings:
        # Get unique values from both train and test sets
        unique_values = pd.concat([X_lgbm[col], test_lgbm[col]]).unique()
        # Create mapping dictionary with a default value for unseen categories
        category_mappings[col] = {val: idx for idx, val in enumerate(unique_values)}
        category_mappings[col]['_unknown'] = len(unique_values)
    
    # Transform using the mapping, with fallback to unknown category
    X_lgbm[col] = X_lgbm[col].map(lambda x: category_mappings[col].get(x, category_mappings[col]['_unknown']))
    test_lgbm[col] = test_lgbm[col].map(lambda x: category_mappings[col].get(x, category_mappings[col]['_unknown']))

catboost_model.fit(X_catboost, y, verbose=100)
lgbm_model.fit(X_lgbm, y)

# Make predictions
catboost_pred = catboost_model.predict(test_catboost)
lgbm_pred = lgbm_model.predict(test_lgbm)

# Blend predictions
blended_pred = 0.7 * catboost_pred + 0.3 * lgbm_pred
test_pred = np.expm1(blended_pred)  # Reverse log transformation

# Create submission file
submission = pd.DataFrame({
    'Id': test['Id'],
    'Price': test_pred
})

submission.to_csv('submission.csv', index=False)
print('Submission file created successfully!')
import pandas as pd
import joblib

# Load pre-fitted transformers from training
# You should save these in your notebook using joblib.dump()
try:
    scaler = joblib.load('models/scaler.pkl')
    imputer = joblib.load('models/imputer.pkl')
    encoder = joblib.load('models/encoder.pkl')
except:
    print("Warning: Model assets not found. Ensure .pkl files exist in 'models/'.")

NUMERIC_COLS = ['age', 'cigsPerDay', 'totChol', 'sysBP', 'BMI', 'glucose']
DROP_COLS = ['prevalentStroke', 'diabetes', 'BPMeds']

def preprocess_input(data_dict):
    """
    Transforms a single prediction request (JSON/Dict) into a format 
    the Random Forest model understands.
    """
    # 1. Convert to DataFrame
    df = pd.DataFrame([data_dict])
    
    # 2. Drop low-value columns
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    # 3. Categorical Mapping (Gender)
    # Expects raw strings like 'Male'/'Female' or 'uneducated'
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1})

    # 4. Ordinal Encoding (Education)
    if 'education' in df.columns:
        # Note: If education is missing, notebook used mode. 
        # For a single prediction, we use the pre-fitted encoder.
        df[['education']] = encoder.transform(df[['education']])

    # 5. Imputation (KNN)
    # We use the fitted imputer to fill missing values based on training patterns
    df_imputed = pd.DataFrame(
        imputer.transform(df),
        columns=df.columns,
        index=df.index
    )

    # 6. Feature Scaling
    df_imputed[NUMERIC_COLS] = scaler.transform(df_imputed[NUMERIC_COLS])

    return df_imputed
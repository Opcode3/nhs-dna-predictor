import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import shap
import joblib

df = pd.read_csv("data/nhs_appointments_Aug_2024_2025_with_imd.csv")

# Target
df['DNA'] = (df['APPT_STATUS'] == 'DNA').astype(int)

# Features (exactly what you said: only known at booking time)
cat_features = ['SUB_ICB_LOCATION_CODE', 'ICB_ONS_CODE', 'REGION_ONS_CODE',
                'HCP_TYPE', 'APPT_MODE']
num_features = ['TIME_BETWEEN_BOOK_AND_APPT', 'IMD_Decile_ICB',
                'Appointment_Month', 'Appointment_Weekday', 'Appointment_Week']  # add these in feature eng

# Simple feature engineering
df['Appointment_Date'] = pd.to_datetime(df['Appointment_Date'])
df['Appointment_Month'] = df['Appointment_Date'].dt.month
df['Appointment_Weekday'] = df['Appointment_Date'].dt.weekday
df['Appointment_Week'] = df['Appointment_Date'].dt.isocalendar().week

for col in cat_features:
    df[col] = df[col].astype('category')

X = df[cat_features + num_features + ['COUNT_OF_APPOINTMENTS']]  # weight by count later
y = df['DNA']

# Temporal split (very important!)
train = df[df['Appointment_Date'] < '2025-01-01']
test  = df[df['Appointment_Date'] >= '2025-01-01']

X_train, X_test = train.drop(['DNA','APPT_STATUS'], axis=1), test.drop(['DNA','APPT_STATUS'], axis=1)
y_train, y_test = train['DNA'], test['DNA']
w_train = train['COUNT_OF_APPOINTMENTS']
w_test  = test['COUNT_OF_APPOINTMENTS']

model = xgb.XGBClassifier(
    n_estimators=600,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight = (w_train*y_train).sum() / y_train.sum(),  # rough imbalance
    enable_categorical=True,
    tree_method='hist',
    random_state=42
)

model.fit(X_train, y_train, sample_weight=w_train, eval_set=[(X_test, y_test)], 
           sample_weight_eval_set=[w_test], verbose=50)

print("Test AUC:", roc_auc_score(y_test, model.predict_proba(X_test)[:,1], sample_weight=w_test))

# Save everything
model.save_model("model/xgb_dna_model.json")

# SHAP (on a sample — too big otherwise)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test.sample(100_000, random_state=42))
joblib.dump(explainer, "model/explainer.pkl")
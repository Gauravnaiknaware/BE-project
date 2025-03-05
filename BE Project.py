#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[38]:


import numpy as np


# In[2]:


import sys
print(sys.version)


# In[3]:


get_ipython().system('pip install xgboost --user')


# In[4]:


import xgboost as xgb


# In[5]:


from sklearn.model_selection import train_test_split


# In[6]:


from sklearn.metrics import mean_squared_error, r2_score


# In[7]:


file_path="C:/Users/Gaurav/OneDrive/Desktop/GDSC_DATASET.csv"


# In[8]:


df=pd.read_csv(file_path)


# In[9]:


df.head()


# In[10]:


from sklearn.preprocessing import LabelEncoder


# In[11]:


categorical_cols=df.select_dtypes(include=['object']).columns


# In[12]:


label_encoders={}
       
       
for col in categorical_cols:
     le=LabelEncoder()
     df[col]=le.fit_transform(df[col].astype(str))
     label_encoders[col]=le


# In[13]:


df.fillna(df.median(numeric_only=True),inplace=True)


# In[14]:


print(df.columns)


# In[15]:


# Selecting features (X) and target variable (y)
target = "LN_IC50"  # Dependent variable
features = ["DRUG_ID", "GDSC Tissue descriptor 1", "GDSC Tissue descriptor 2", 
            "Cancer Type (matching TCGA label)", "Microsatellite instability Status (MSI)"]


# In[16]:


print(df.columns)


# In[17]:


X = df.drop(columns=["TARGET"])  # Features (remove the target column)
y = df["TARGET"]  # Target variable


# In[18]:


df.columns = df.columns.str.strip()


# In[19]:


if "Microsatellite instability Status (MSI)" in df.columns:
    print("Column exists!")
else:
    print("Column is missing! Check spelling.")


# In[20]:


df["Microsatellite instability Status (MSI)"] = df["Microsatellite instability Status (MSI)"].fillna("Unknown")


# In[21]:


print(features)


# In[22]:


if "Microsatellite instability Status (MSI)" in features:
    features.remove("Microsatellite instability Status (MSI)")


# In[23]:


print([col for col in df.columns if "Microsatellite" in col])


# In[24]:


df.columns = df.columns.str.strip()


# In[25]:


print(features)


# In[26]:


for col in df.columns:
    if "Microsatellite" in col:
        print(col)


# In[27]:


features = ['DRUG_ID', 'GDSC Tissue descriptor 1', 'GDSC Tissue descriptor 2', 
            'Cancer Type (matching TCGA label)', 'Microsatellite instability Status (MSI)']


# In[28]:


features.remove("Microsatellite instability Status (MSI)")
print(features)


# In[29]:


df.columns = df.columns.str.strip()


# In[30]:


print(df.columns.tolist())


# In[ ]:





# In[31]:


df = df[features + [target]]


# In[32]:


X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[33]:


print(X_train)


# In[34]:


xgb_regressor = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
xgb_regressor.fit(X_train, y_train)


# In[35]:


print(y_train)


# In[36]:


# Predictions
y_pred = xgb_regressor.predict(X_test)


# In[39]:


# Model Evaluation
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)


# In[40]:


print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R² Score: {r2:.4f}")


# In[41]:


import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, color="blue", alpha=0.5, label="Predicted")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle="--", color="red", label="Perfect Fit")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted (XGBoost)")
plt.legend()
plt.show()


# In[42]:


residuals = y_test - y_pred

plt.figure(figsize=(8,6))
plt.hist(residuals, bins=30, color="purple", alpha=0.7)
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.title("Residual Histogram")
plt.show()


# In[43]:


from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R-squared (R²): {r2:.4f}")


# In[ ]:





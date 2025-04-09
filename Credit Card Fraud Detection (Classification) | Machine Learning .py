#!/usr/bin/env python
# coding: utf-8

# In[24]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt


# In[2]:


df= pd.read_csv('/Users/abusayed/Downloads/creditcard.csv')


# In[3]:


df.head()


# In[8]:


# Display basic information and the first few rows
df_info = df.info()
df_head = df.head()



# In[11]:


df_description = df.describe()


# In[13]:


df_description


# In[14]:


# Check class distribution
class_distribution = df['Class'].value_counts()


# In[15]:


class_distribution


# In[16]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


# In[17]:


# Separate features and target
X = df.drop("Class", axis=1)
y = df["Class"]


# In[18]:


# Scale the 'Amount' and 'Time' features (others are already PCA-transformed)
scaler = StandardScaler()
X[["Time", "Amount"]] = scaler.fit_transform(X[["Time", "Amount"]])


# In[19]:


# Train-test split with stratification to maintain class balance
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)


# In[20]:


# Train Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


# In[21]:


# Predictions
y_pred = model.predict(X_test)


# In[23]:


# Evaluation
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, output_dict=True)
conf_matrix, class_report


# In[25]:


# Predict probabilities for class 1 (fraud)
y_probs = model.predict_proba(X_test)[:, 1]


# In[26]:


# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
# Calculate AUC
roc_auc = roc_auc_score(y_test, y_probs)



# In[29]:


# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"Logistic Regression (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.show()




# In[ ]:





# In[ ]:





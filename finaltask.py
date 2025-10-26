#C:/ProgramData/Anaconda3/python.exe "C:/Users/varsh/Desktop/final task/finaltask.py"
# import pandas as pd

# # Use full path to avoid FileNotFoundError
# df = pd.read_csv(r"C:\Users\varsh\Desktop\final task\ecommerce_returns_synthetic_data.csv")

# print(df.head())

# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # -------------------------
# # 1️⃣ Load the dataset
# # -------------------------
# df = pd.read_csv(r"C:\Users\varsh\Desktop\final task\ecommerce_returns_synthetic_data.csv")

# # -------------------------
# # 2️⃣ Basic info & overview
# # -------------------------
# print("===== Dataset Info =====")
# print(df.info())

# print("\n===== First 5 Rows =====")
# print(df.head())

# print("\n===== Column Names =====")
# print(df.columns)

# print("\n===== Missing Values =====")
# print(df.isnull().sum())

# # -------------------------
# # 3️⃣ Return analysis
# # -------------------------
# if 'Return' in df.columns:
#     print("\n===== Return Counts =====")
#     print(df['Return'].value_counts())

#     return_rate = df['Return'].value_counts(normalize=True) * 100
#     print("\nReturn Rate (%)")
#     print(return_rate)

# # -------------------------
# # 4️⃣ Payment & Shipping methods
# # -------------------------
# if 'Payment_Method' in df.columns:
#     print("\n===== Payment Method Counts =====")
#     print(df['Payment_Method'].value_counts())

# if 'Shipping_Method' in df.columns:
#     print("\n===== Shipping Method Counts =====")
#     print(df['Shipping_Method'].value_counts())

# # -------------------------
# # 5️⃣ Discount analysis
# # -------------------------
# if 'Discount_Applied' in df.columns:
#     print("\n===== Discount Applied Statistics =====")
#     print(df['Discount_Applied'].describe())

# # -------------------------
# # 6️⃣ Visualizations
# # -------------------------
# sns.set(style="whitegrid")

# # Return counts
# if 'Return' in df.columns:
#     plt.figure(figsize=(6,4))
#     sns.countplot(x='Return', data=df)
#     plt.title('Return Counts')
#     plt.show()

# # Return by Payment Method
# if 'Return' in df.columns and 'Payment_Method' in df.columns:
#     plt.figure(figsize=(8,5))
#     sns.countplot(x='Payment_Method', hue='Return', data=df)
#     plt.title('Returns by Payment Method')
#     plt.xticks(rotation=45)
#     plt.show()

# # Return by Shipping Method
# if 'Return' in df.columns and 'Shipping_Method' in df.columns:
#     plt.figure(figsize=(8,5))
#     sns.countplot(x='Shipping_Method', hue='Return', data=df)
#     plt.title('Returns by Shipping Method')
#     plt.xticks(rotation=45)
#     plt.show()

# df['Return_Status'].value_counts(normalize=True) * 100
# pd.crosstab(df['Payment_Method'], df['Return_Status'], normalize='index') * 100
# pd.crosstab(df['Shipping_Method'], df['Return_Status'], normalize='index') * 100
# top_returned_products = df[df['Return_Status']=='Returned']['Product_ID'].value_counts().head(10)
# top_returned_categories = df[df['Return_Status']=='Returned']['Product_Category'].value_counts().head(10)



# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # -------------------------
# # 1️⃣ Load the dataset
# # -------------------------
# df = pd.read_csv(r"C:\Users\varsh\Desktop\final task\ecommerce_returns_synthetic_data.csv")

# # -------------------------
# # 2️⃣ Basic overview
# # -------------------------
# print("===== Dataset Info =====")
# print(df.info())

# print("\n===== First 5 Rows =====")
# print(df.head())

# print("\n===== Missing Values =====")
# print(df.isnull().sum())

# # -------------------------
# # 3️⃣ Return Analysis
# # -------------------------
# print("\n===== Return Status Counts =====")
# return_counts = df['Return_Status'].value_counts()
# print(return_counts)

# print("\n===== Return Rate (%) =====")
# return_rate = df['Return_Status'].value_counts(normalize=True) * 100
# print(return_rate)

# # -------------------------
# # 4️⃣ Returns by Payment & Shipping
# # -------------------------
# print("\n===== Returns by Payment Method (%) =====")
# payment_returns = pd.crosstab(df['Payment_Method'], df['Return_Status'], normalize='index')*100
# print(payment_returns)

# print("\n===== Returns by Shipping Method (%) =====")
# shipping_returns = pd.crosstab(df['Shipping_Method'], df['Return_Status'], normalize='index')*100
# print(shipping_returns)

# # -------------------------
# # 5️⃣ Top Returned Products & Categories
# # -------------------------
# top_products = df[df['Return_Status']=='Returned']['Product_ID'].value_counts().head(10)
# top_categories = df[df['Return_Status']=='Returned']['Product_Category'].value_counts().head(10)

# print("\n===== Top 10 Returned Products =====")
# print(top_products)

# print("\n===== Top 10 Returned Categories =====")
# print(top_categories)

# # -------------------------
# # 6️⃣ Visualizations
# # -------------------------
# sns.set(style="whitegrid")

# # Overall return counts
# plt.figure(figsize=(6,4))
# sns.countplot(x='Return_Status', data=df)
# plt.title('Return Counts')
# plt.show()

# # Returns by Payment Method
# plt.figure(figsize=(8,5))
# sns.countplot(x='Payment_Method', hue='Return_Status', data=df)
# plt.title('Returns by Payment Method')
# plt.xticks(rotation=45)
# plt.show()

# # Returns by Shipping Method
# plt.figure(figsize=(8,5))
# sns.countplot(x='Shipping_Method', hue='Return_Status', data=df)
# plt.title('Returns by Shipping Method')
# plt.xticks(rotation=45)
# plt.show()

# # Discount Applied vs Return Status
# plt.figure(figsize=(8,5))
# sns.boxplot(x='Return_Status', y='Discount_Applied', data=df)
# plt.title('Discounts Applied vs Return Status')
# plt.show()

# # Top 10 Returned Products Bar Chart
# plt.figure(figsize=(10,5))
# top_products.plot(kind='bar', color='orange')
# plt.title('Top 10 Returned Products')
# plt.xlabel('Product_ID')
# plt.ylabel('Number of Returns')
# plt.show()

# # Top 10 Returned Categories Bar Chart
# plt.figure(figsize=(8,5))
# top_categories.plot(kind='bar', color='green')
# plt.title('Top 10 Returned Categories')
# plt.xlabel('Category')
# plt.ylabel('Number of Returns')
# plt.show()

# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # -------------------------
# # 1️⃣ Load the dataset
# # -------------------------
# df = pd.read_csv(r"C:\Users\varsh\Desktop\final task\ecommerce_returns_synthetic_data.csv")

# # -------------------------
# # 2️⃣ Basic Overview
# # -------------------------
# print("===== Dataset Info =====")
# print(df.info())
# print("\n===== First 5 Rows =====")
# print(df.head())
# print("\n===== Missing Values =====")
# print(df.isnull().sum())

# # -------------------------
# # 3️⃣ Return Analysis
# # -------------------------
# print("\n===== Return Status Counts =====")
# return_counts = df['Return_Status'].value_counts()
# print(return_counts)

# print("\n===== Return Rate (%) =====")
# return_rate = df['Return_Status'].value_counts(normalize=True) * 100
# print(return_rate)

# # -------------------------
# # 4️⃣ Returns by Payment & Shipping
# # -------------------------
# print("\n===== Returns by Payment Method (%) =====")
# print(pd.crosstab(df['Payment_Method'], df['Return_Status'], normalize='index')*100)

# print("\n===== Returns by Shipping Method (%) =====")
# print(pd.crosstab(df['Shipping_Method'], df['Return_Status'], normalize='index')*100)

# # -------------------------
# # 5️⃣ Top Returned Products & Categories
# # -------------------------
# top_products = df[df['Return_Status']=='Returned']['Product_ID'].value_counts().head(10)
# top_categories = df[df['Return_Status']=='Returned']['Product_Category'].value_counts().head(10)

# print("\n===== Top 10 Returned Products =====")
# print(top_products)

# print("\n===== Top 10 Returned Categories =====")
# print(top_categories)

# # -------------------------
# # 6️⃣ Return Timing Analysis
# # -------------------------
# returned_orders = df[df['Return_Status']=='Returned']
# print("\n===== Days to Return Statistics =====")
# print(returned_orders['Days_to_Return'].describe())

# plt.figure(figsize=(8,5))
# sns.histplot(returned_orders['Days_to_Return'], bins=30, kde=True)
# plt.title('Distribution of Return Times')
# plt.xlabel('Days to Return')
# plt.ylabel('Number of Returns')
# plt.show()

# # -------------------------
# # 7️⃣ Returns by User Demographics
# # -------------------------
# # Gender
# if 'User_Gender' in df.columns:
#     plt.figure(figsize=(6,4))
#     sns.countplot(x='User_Gender', hue='Return_Status', data=df)
#     plt.title('Returns by Gender')
#     plt.show()

# # Age
# if 'User_Age' in df.columns:
#     plt.figure(figsize=(8,5))
#     sns.histplot(data=returned_orders, x='User_Age', bins=20)
#     plt.title('Returns by Age')
#     plt.xlabel('User Age')
#     plt.ylabel('Number of Returns')
#     plt.show()

# # Location
# if 'User_Location' in df.columns:
#     top_locations = returned_orders['User_Location'].value_counts().head(10)
#     plt.figure(figsize=(10,5))
#     top_locations.plot(kind='bar', color='purple')
#     plt.title('Top 10 Locations with Returns')
#     plt.xlabel('Location')
#     plt.ylabel('Number of Returns')
#     plt.show()

# # -------------------------
# # 8️⃣ Discount Analysis for Returns
# # -------------------------
# plt.figure(figsize=(8,5))
# sns.boxplot(x='Return_Status', y='Discount_Applied', data=df)
# plt.title('Discounts Applied vs Return Status')
# plt.show()

# # -------------------------
# # 9️⃣ Visualizations Recap
# # -------------------------
# # Overall Return Counts
# plt.figure(figsize=(6,4))
# sns.countplot(x='Return_Status', data=df)
# plt.title('Overall Return Counts')
# plt.show()

# # Returns by Payment Method
# plt.figure(figsize=(8,5))
# sns.countplot(x='Payment_Method', hue='Return_Status', data=df)
# plt.title('Returns by Payment Method')
# plt.xticks(rotation=45)
# plt.show()

# # Returns by Shipping Method
# plt.figure(figsize=(8,5))
# sns.countplot(x='Shipping_Method', hue='Return_Status', data=df)
# plt.title('Returns by Shipping Method')
# plt.xticks(rotation=45)
# plt.show()

# # Top Returned Products
# plt.figure(figsize=(10,5))
# top_products.plot(kind='bar', color='orange')
# plt.title('Top 10 Returned Products')
# plt.xlabel('Product_ID')
# plt.ylabel('Number of Returns')
# plt.show()

# # Top Returned Categories
# plt.figure(figsize=(8,5))
# top_categories.plot(kind='bar', color='green')
# plt.title('Top 10 Returned Categories')
# plt.xlabel('Category')
# plt.ylabel('Number of Returns')
# plt.show()


# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import classification_report, roc_auc_score

# # -------------------------
# # 1️⃣ Load dataset
# # -------------------------
# df = pd.read_csv(r"C:\Users\varsh\Desktop\final task\ecommerce_returns_synthetic_data.csv")

# # -------------------------
# # 2️⃣ Data Cleaning / Preparation
# # -------------------------
# # Fill missing numerical values if needed
# df['Discount_Applied'] = df['Discount_Applied'].fillna(0)
# df['Order_Quantity'] = df['Order_Quantity'].fillna(1)
# df['Product_Price'] = df['Product_Price'].fillna(df['Product_Price'].median())

# # Fill missing categorical values
# df['Payment_Method'] = df['Payment_Method'].fillna('Unknown')
# df['Shipping_Method'] = df['Shipping_Method'].fillna('Standard')
# df['Product_Category'] = df['Product_Category'].fillna('Other')

# # Encode target variable
# df['Return_Flag'] = df['Return_Status'].apply(lambda x: 1 if x=='Returned' else 0)

# # -------------------------
# # 3️⃣ Encode Categorical Variables
# # -------------------------
# categorical_cols = ['Payment_Method', 'Shipping_Method', 'Product_Category', 'User_Gender']
# for col in categorical_cols:
#     if col in df.columns:
#         le = LabelEncoder()
#         df[col+'_enc'] = le.fit_transform(df[col])

# # -------------------------
# # 4️⃣ Define Features and Target
# # -------------------------
# feature_cols = ['Product_Price', 'Order_Quantity', 'Discount_Applied']
# # Add encoded categorical columns if exist
# for col in categorical_cols:
#     if col+'_enc' in df.columns:
#         feature_cols.append(col+'_enc')

# X = df[feature_cols]
# y = df['Return_Flag']

# # -------------------------
# # 5️⃣ Train/Test Split
# # -------------------------
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # -------------------------
# # 6️⃣ Logistic Regression Model
# # -------------------------
# model = LogisticRegression(max_iter=1000)
# model.fit(X_train, y_train)

# # -------------------------
# # 7️⃣ Predictions and Evaluation
# # -------------------------
# y_pred = model.predict(X_test)
# y_prob = model.predict_proba(X_test)[:,1]

# print("===== Classification Report =====")
# print(classification_report(y_test, y_pred))

# print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))

# # -------------------------
# # 8️⃣ Add Return Probability to Dataset
# # -------------------------
# df['Return_Prob'] = model.predict_proba(X)[:,1]

# # -------------------------
# # 9️⃣ Export High-Risk Products (Return Prob > 0.7)
# # -------------------------
# high_risk = df[df['Return_Prob'] > 0.7]
# high_risk.to_csv(r"C:\Users\varsh\Desktop\final task\high_risk_products.csv", index=False)
# print(f"High-risk products exported: {len(high_risk)} rows")


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

# -------------------------
# 1️⃣ Load dataset
# -------------------------
df = pd.read_csv(r"C:\Users\varsh\Desktop\final task\ecommerce_returns_synthetic_data.csv")

# -------------------------
# 2️⃣ Data Cleaning / Preparation
# -------------------------
# Fill missing values
df['Discount_Applied'] = df['Discount_Applied'].fillna(0)
df['Order_Quantity'] = df['Order_Quantity'].fillna(1)
df['Product_Price'] = df['Product_Price'].fillna(df['Product_Price'].median())
df['Payment_Method'] = df['Payment_Method'].fillna('Unknown')
df['Shipping_Method'] = df['Shipping_Method'].fillna('Standard')
df['Product_Category'] = df['Product_Category'].fillna('Other')
df['User_Gender'] = df['User_Gender'].fillna('Unknown')

# Target variable
df['Return_Flag'] = df['Return_Status'].apply(lambda x: 1 if x=='Returned' else 0)

# Feature: Order amount
df['Order_Amount'] = df['Product_Price'] * df['Order_Quantity']

# -------------------------
# 3️⃣ One-hot Encoding for categorical variables
# -------------------------
categorical_cols = ['Payment_Method', 'Shipping_Method', 'Product_Category', 'User_Gender']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# -------------------------
# 4️⃣ Define Features and Target
# -------------------------
feature_cols = ['Product_Price', 'Order_Quantity', 'Discount_Applied', 'Order_Amount'] + \
               [col for col in df.columns if any(cat in col for cat in categorical_cols)]
X = df[feature_cols]
y = df['Return_Flag']

# -------------------------
# 5️⃣ Train/Test Split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# -------------------------
# 6️⃣ Logistic Regression with balanced classes
# -------------------------
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train, y_train)

# -------------------------
# 7️⃣ Predictions and Evaluation
# -------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

print("===== Classification Report =====")
print(classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))

# -------------------------
# 8️⃣ Add Return Probability to Dataset
# -------------------------
df['Return_Prob'] = model.predict_proba(X)[:,1]

# -------------------------
# 9️⃣ Export High-Risk Products (Return Prob > 0.5)
# -------------------------
high_risk = df[df['Return_Prob'] > 0.5]
high_risk.to_csv(r"C:\Users\varsh\Desktop\final task\high_risk_products.csv", index=False)
print(f"High-risk products exported: {len(high_risk)} rows")

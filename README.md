# final-task-
E-commerce Return Prediction and Analysis Project
This project focuses on analyzing e-commerce return data to understand key factors driving product returns and building a predictive model to identify high-risk orders before they are shipped or processed.

1. Data Analysis and Visualization (EDA)
The initial exploratory data analysis (EDA) provided key insights into the distribution of returns across various dimensions, as detailed in the attached PDF visualizations (pdf24_converted.pdf, pdf24_converted (1).pdf).

Key Findings:
Return Status Balance: The dataset shows a nearly balanced distribution between "Returned" and "Not Returned" orders (approximately 5,000 counts each).

Demographics: Returns are observed across all age groups (20s to 70s), with slightly higher counts in the early 30s and early 60s. Returns by gender are also roughly equal.

Product: The Clothing category has the highest number of returns, followed by Books, Electronics, Home, and Toys.

Discounts: The analysis suggests that the discount percentage applied is not a strong predictor of returns, as the distribution is nearly identical for both returned and non-returned orders (median discount around 25% for both).

Location: City43 is identified as the location with the highest absolute count of returns.

2. Predictive Modeling (Logistic Regression)
The finaltask.py script implements a machine learning pipeline to predict the probability of an order being returned.

Model Workflow:
Data Preprocessing: Missing values in numerical features (Discount_Applied, Order_Quantity, Product_Price) were imputed.

Feature Engineering: A target variable, Return_Flag (1 for Returned, 0 otherwise), was created. The Order_Amount was calculated (Product_Price * Order_Quantity).

Encoding: Categorical variables (Payment_Method, Shipping_Method, Product_Category, User_Gender) were converted using One-Hot Encoding.

Model Training: A Logistic Regression model was trained using a stratified 80/20 train/test split. The parameter class_weight='balanced' was used to ensure the model performs well on both classes.

High-Risk Identification:
The model calculates a Return_Prob for every order.

Orders where Return_Prob > 0.5 are flagged as "high-risk."

These high-risk orders are then exported to a file named high_risk_products.csv. This output can be used by the business to enable proactive intervention, such as targeted customer service or quality checks, to potentially reduce the actual return rate.

Code and Dependencies:
The main script uses standard data science libraries:

pandas for data manipulation.

sklearn (LogisticRegression, train_test_split, classification_report, roc_auc_score) for modeling and evaluation.

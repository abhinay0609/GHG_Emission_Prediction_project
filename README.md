# GHG_Emission_Prediction_Project
This is a 4 week Edunet-Shell Skills4Future AICTE Internship focused on Green Skills & AI through project based learning.

**Project Title:** Grrenhouse Gas Emission Prediction

**Problem Statement:**
You have annual supply chain emission data from 2010–2016 categorized into industries and commodities. The goal is to develop a regression model that can predict the Supply Chain Emission Factors with Margins based on descriptive and quality metrics (substance, unit, reliability, temporal/geographical/technological/data collection correlations, etc.).

**Project Goal:**
To analyze and predict greenhouse gas (GHG) emissions from various U.S. industries and commodities using the official dataset from data.gov.

**Source:**
Supply Chain Greenhouse Gas Emission Factors

**Tools:**
Python, Pandas, Scikit-learn, Matplotlib, Seaborn

**Steps:**

Step 1: Import Required Libraries

Step 2: Load Dataset 

Step 3: Data Preprocessing (EDA+Cleaning+Encoding) 

Step 4: Training 

Step 5: Prediction and Evaluation 

Step 6: Hyperparameter Tuning 

Step 7: Comparative Study and Selecting the Best model 


## 📅 Week 1: Understanding the Problem & Dataset

- 🔍 Explored multi-sheet Excel dataset (2010–2016) combining **Industry** and **Commodity** data.
- 🧠 Learned to load, clean, and structure large datasets using **Pandas**, merge sheets based on schema similarity.
- 📊 Outcome: One clean, combined DataFrame with meaningful features ready for preprocessing.

## 🧮 Week 2: Data Preprocessing & Exploratory Data Analysis (EDA)

- 🧹 Handled missing values, dropped unnecessary columns, and explored data types.
- 📈 Performed **univariate & multivariate analysis** using **Seaborn heatmaps, histograms**, and **correlation matrix**.
- 🔍 Investigated **normalization, skewness, and feature scaling** to prepare data for training.
- ✅ Split data into `X_train`, `X_test`, `y_train`, and `y_test` after scaling with `StandardScaler`.

## Final Week 3: Model Building, Evaluation & Deployment

- 🤖 Trained both **Linear Regression** and **Random Forest** models. Also performed **GridSearchCV tuning**.
- 📊 Achieved exceptional results:  
  - **Linear Regression R² Score**: ` 0.999999`  
  - **Random Forest R² Score**: `0.999380`  
  - **RMSE** (Test): Very low, indicating strong predictive performance.
- ⚖️ Selected **Linear Regression** as final model due to simplicity, stability, and performance.
- 🌐 Deployed the model using **Streamlit**; includes input preprocessing and `.pkl` loading logic.

## 🌟 Highlights

- ✅ **End-to-End ML Pipeline**: From raw dataset cleaning to deployment with Streamlit.
- 📊 **Model Evaluation Metrics**: RMSE, R² Score, correlation heatmaps, and feature scaling all performed.
- 💼 **Business Impact**: Enables accurate forecasting of supply chain GHG emissions — crucial for sustainability planning.
- 🚀 **Deployment Ready**: Streamlit app built; model & scaler zipped.


   

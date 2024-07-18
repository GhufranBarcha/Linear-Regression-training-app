import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

# Set page layout to wide
st.set_page_config(layout="wide")

# Title
st.title("Linear Regression Model Training")

# Sidebar for file upload and drop columns
st.sidebar.title("Upload the CSV")

# File uploader in the sidebar
uploaded_file = st.sidebar.file_uploader("Click Here", type=["csv"])

if uploaded_file is not None:
    # Read the file
    df = pd.read_csv(uploaded_file)
    
    # Keep only numerical columns
    df = df.select_dtypes(exclude=['object'])
    
    # Drop NaN values
    df = df.dropna()
    
    # Display processed DataFrame head in the main content area
    st.write("Processed DataFrame Head (Numerical Columns):")
    st.dataframe(df, use_container_width=True, height=230)
    
    # Select only numerical columns
    numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Sidebar selectors for dependent and independent variables
    st.sidebar.subheader("Variable Selection")
    
    # Select dependent variable
    dependent_var = st.sidebar.selectbox("Select dependent variable:", numerical_columns)
    
    if dependent_var:
        # Filter out the dependent variable from the list of numerical columns for independent variable selection
        independent_columns = [col for col in numerical_columns if col != dependent_var]
        
        # Multiselect for selecting independent variables with all options initially selected
        independent_vars = st.sidebar.multiselect("Select independent variables:", independent_columns, default=independent_columns)
        
        if independent_vars:
            X = df[independent_vars]
            y = df[dependent_var]
            
            # Scale the independent variables
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Polynomial features option
            use_poly = st.sidebar.checkbox("Use Polynomial Regression")
            degree = 2
            if use_poly:
                degree = st.sidebar.slider("Select Polynomial Degree", 2, 5, 2)
                poly = PolynomialFeatures(degree)
                X_scaled = poly.fit_transform(X_scaled)
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
            
            # Train the model
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # "Predict" button in the sidebar
            if st.sidebar.button("Predict"):
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mape = mean_absolute_percentage_error(y_test, y_pred)
                
                # Round metrics to two decimal places
                r2 = round(r2, 3)
                mae = round(mae, 2)
                mse = round(mse, 2)
                rmse = round(rmse, 2)
                mape = round(mape, 2)
                
                # Display metrics in the main content area
                st.subheader("Model Performance Metrics:")
                col1, col2, col3, col4, col5 = st.columns(5)
                
                col1.metric("R2-Score", f"{r2 * 100} %")
                # col2.metric("MAPE", mape)
                col2.metric("MAE", mae)
                col3.metric("RMSE", rmse)
                col4.metric("MSE", mse)
                
                if not use_poly:
                                                    # Example coefficients and feature names
                    coef = np.array(model.coef_).reshape(1, -1)
                    columns = independent_vars

                    # Create a DataFrame
                    s = pd.DataFrame(coef, columns=independent_vars)
                  

                    # Take absolute values of coefficients
                    s_abs = s.abs()

                    # Transpose and reset index for sorting
                    df_t = s_abs.T.reset_index()

                    # Sort by absolute coefficient in descending order
                    df_sorted = df_t.sort_values(by=0, ascending=False)
                    
                    for i in df_sorted["index"]:
                         print()
                    
                    
                    st.write("### All Variable Impacts:")

                    cols = st.columns([2, 2, 2, 2])
                    # Display variable impacts in multiple columns
                    for ind, i in enumerate(df_sorted["index"]):
                        col = cols[ind % len(cols)]
                        col.success(f"{i}: {round(s.T.loc[i][0], 2)}")    
                        
                        
                        
                
                

                
        else:
            st.sidebar.write("Please select at least one independent variable.")
else:
    # Introductory text when no CSV is uploaded
    st.write("""
Welcome to the **Linear Regression Model Training App**

This tool allows you to:

- Upload a CSV file.
- Select the dependent and independent variables.
- Train a linear regression model on your data.
- View various performance metrics of the trained model.

### Getting Started

To begin, please upload a CSV file using the sidebar on the left.
""")















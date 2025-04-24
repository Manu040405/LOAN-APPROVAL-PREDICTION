import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time
import os

# Import custom modules
from loan_model import LoanPredictionModel
from utils import (
    calculate_emi, 
    get_amortization_schedule, 
    save_prediction_history, 
    get_prediction_history,
    get_dataset_summary,
    plot_feature_importance,
    plot_approval_distribution
)

# Set page config
st.set_page_config(
    page_title="Loan Approval Prediction System",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define CSS for custom styling (minimal, using Streamlit's built-in components)
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
    }
    .info-box {
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv("LOAN.csv")

# Initialize the model
@st.cache_resource
def initialize_model():
    model = LoanPredictionModel()
    model.train("LOAN.csv")
    return model

# Main function
def main():
    # App title
    st.markdown("<h1 class='main-header'>🏦 Loan Approval Prediction System</h1>", unsafe_allow_html=True)
    
    # Initialize session state for navigation
    if 'current_page' not in st.session_state:
        st.session_state['current_page'] = 'home'
    
    # Load data and model
    df = load_data()
    model = initialize_model()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    
    # Navigation buttons
    if st.sidebar.button("🏠 Home", use_container_width=True):
        st.session_state['current_page'] = 'home'
    if st.sidebar.button("🔮 Loan Prediction", use_container_width=True):
        st.session_state['current_page'] = 'prediction'
    if st.sidebar.button("🧮 EMI Calculator", use_container_width=True):
        st.session_state['current_page'] = 'emi'
    if st.sidebar.button("📊 Data Visualization", use_container_width=True):
        st.session_state['current_page'] = 'visualization'
    if st.sidebar.button("📝 Prediction History", use_container_width=True):
        st.session_state['current_page'] = 'history'
    if st.sidebar.button("🔍 Data Explorer", use_container_width=True):
        st.session_state['current_page'] = 'explorer'
    
    # Display the appropriate page
    if st.session_state['current_page'] == 'home':
        home_page(df, model)
    elif st.session_state['current_page'] == 'prediction':
        prediction_page(model, df)
    elif st.session_state['current_page'] == 'emi':
        emi_calculator_page()
    elif st.session_state['current_page'] == 'visualization':
        visualization_page(df, model)
    elif st.session_state['current_page'] == 'history':
        history_page()
    elif st.session_state['current_page'] == 'explorer':
        data_explorer_page(df)

def home_page(df, model):
    """Home page with summary information and key stats"""
    # Get dataset summary
    summary = get_dataset_summary(df)
    
    # Display dataset summary
    st.markdown("<h2 class='sub-header'>📊 Loan Application Dataset Overview</h2>", unsafe_allow_html=True)
    
    # Show key metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Applications", f"{summary['total_records']}")
    with col2:
        st.metric("Approval Rate", f"{summary['approval_rate']*100:.1f}%")
    with col3:
        st.metric("Model Accuracy", f"{model.accuracy*100:.1f}%")
    
    # More stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Avg. Loan Amount", f"₹{summary['avg_loan_amount']*1000:.2f}")
    with col2:
        st.metric("Avg. Applicant Income", f"₹{summary['avg_applicant_income']:.2f}")
    with col3:
        st.metric("Avg. Loan Term", f"{summary['avg_loan_term']:.1f} years")
    
    # Application description
    st.markdown("""
    ### Welcome to the Loan Approval Prediction System
    
    This application helps you predict loan approval decisions using machine learning. 
    Key features include:
    
    - **Loan Eligibility Prediction**: Get instant predictions on loan approval chances
    - **EMI Calculator**: Calculate monthly payments and view amortization schedules
    - **Data Visualization**: Explore factors affecting loan approval
    - **Prediction History**: Track all your previous predictions
    - **Data Explorer**: Filter and analyze the loan dataset
    
    Navigate using the sidebar to access different features.
    """)
    
    # Sample visualization - top factors affecting loan approval
    st.markdown("<h2 class='sub-header'>🔑 Key Factors Affecting Loan Approval</h2>", unsafe_allow_html=True)
    
    # Show feature importance
    feature_importance = model.get_feature_importance()
    sorted_features = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(list(sorted_features.keys()), list(sorted_features.values()), color='skyblue')
    ax.set_xlabel('Importance')
    ax.set_title('Top 5 Features Affecting Loan Approval')
    
    # Add values on the bars
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.4f}', ha='left', va='center')
    
    st.pyplot(fig)
    
    # Quick links
    st.markdown("<h2 class='sub-header'>🚀 Quick Actions</h2>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start New Prediction", use_container_width=True):
            st.session_state['current_page'] = 'prediction'
            st.rerun()
    with col2:
        if st.button("Calculate EMI", use_container_width=True):
            st.session_state['current_page'] = 'emi'
            st.rerun()

def prediction_page(model, df):
    """Page for making loan predictions"""
    st.markdown("<h2 class='sub-header'>🔮 Loan Approval Prediction</h2>", unsafe_allow_html=True)
    
    # Create form for input
    with st.form("prediction_form"):
        st.subheader("Enter Applicant Details")
        
        # Personal Information
        col1, col2, col3 = st.columns(3)
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
        with col2:
            married = st.selectbox("Married", ["Yes", "No"])
        with col3:
            dependents = st.selectbox("Number of Dependents", ["0", "1", "2", "3+"])
        
        # Educational and Employment Information
        col1, col2 = st.columns(2)
        with col1:
            education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        with col2:
            self_employed = st.selectbox("Self Employed", ["Yes", "No"])
        
        # Financial Information
        col1, col2 = st.columns(2)
        with col1:
            applicant_income = st.number_input("Applicant Income (₹)", min_value=0, value=5000)
        with col2:
            coapplicant_income = st.number_input("Coapplicant Income (₹)", min_value=0, value=0)
        
        # Loan Details
        col1, col2 = st.columns(2)
        with col1:
            loan_amount = st.number_input("Loan Amount (₹)", min_value=0, value=100000)
        with col2:
            loan_term_years = st.number_input("Loan Amount Term (in years)", 
                                             min_value=1, max_value=40, value=30)
            loan_amount_term = loan_term_years * 12  # convert years to months
        
        # Additional Information
        col1, col2 = st.columns(2)
        with col1:
            credit_history = st.selectbox("Credit History (1 = good, 0 = poor)", [1.0, 0.0])
        with col2:
            property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
        
        # Submit button
        submit_button = st.form_submit_button("Predict Loan Approval")
    
    # Process form submission
    if submit_button:
        with st.spinner("Processing prediction..."):
            # Prepare input data
            input_data = {
                'Gender': gender,
                'Married': married,
                'Dependents': dependents,
                'Education': education,
                'Self_Employed': self_employed,
                'ApplicantIncome': applicant_income,
                'CoapplicantIncome': coapplicant_income,
                'LoanAmount': loan_amount / 1000,  # Convert to thousands
                'Loan_Amount_Term': loan_amount_term,
                'Credit_History': credit_history,
                'Property_Area': property_area,
            }
            
            # Make prediction
            prediction, probability = model.predict(input_data)
            
            # Store prediction in history
            history_data = {
                'input_data': input_data,
                'prediction': prediction,
                'probability': probability,
            }
            save_prediction_history(history_data)
            
            # Display results
            st.subheader("Prediction Results")
            
            # Show probability with progress bar
            st.write(f"Approval Probability: {probability:.2%}")
            st.progress(probability)
            
            # Show prediction
            if prediction == 1:
                st.success("✅ Loan is likely to be APPROVED")
                
                # Calculate EMI for approved loans
                emi = calculate_emi(loan_amount, 8.5, loan_term_years)  # Assume 8.5% interest rate
                st.write(f"Estimated Monthly EMI (at 8.5% interest): ₹{emi:.2f}")
                
                # Option to view detailed EMI calculation
                if st.button("View Detailed EMI Calculation"):
                    st.session_state['current_page'] = 'emi'
                    st.session_state['pre_filled'] = {
                        'loan_amount': loan_amount,
                        'interest_rate': 8.5,
                        'tenure_years': loan_term_years
                    }
                    st.rerun()
            else:
                st.error("❌ Loan is likely to be REJECTED")
                
                # Show key factors that might have led to rejection
                st.subheader("Key factors affecting this prediction:")
                feature_imp = model.get_feature_importance()
                top_factors = dict(sorted(feature_imp.items(), key=lambda x: x[1], reverse=True)[:3])
                
                for factor, importance in top_factors.items():
                    st.write(f"- {factor}: {importance:.4f}")
            
            # Show success message
            st.success("Prediction complete! Results saved to history.")

def emi_calculator_page():
    """EMI calculator page"""
    st.markdown("<h2 class='sub-header'>🧮 Loan EMI Calculator</h2>", unsafe_allow_html=True)
    
    # Pre-fill the form if coming from prediction page
    prefill = st.session_state.get('pre_filled', {})
    loan_amount_default = prefill.get('loan_amount', 100000)
    interest_rate_default = prefill.get('interest_rate', 8.5)
    tenure_years_default = prefill.get('tenure_years', 20)
    
    # Clear pre-filled values after using them
    if 'pre_filled' in st.session_state:
        del st.session_state['pre_filled']
    
    # Create columns for input
    col1, col2, col3 = st.columns(3)
    
    with col1:
        loan_amount = st.number_input("Loan Amount (₹)", 
                                     min_value=1000, 
                                     max_value=10000000, 
                                     value=int(loan_amount_default),
                                     step=1000)
    
    with col2:
        interest_rate = st.number_input("Annual Interest Rate (%)", 
                                       min_value=1.0, 
                                       max_value=30.0, 
                                       value=float(interest_rate_default),
                                       step=0.1)
    
    with col3:
        tenure_years = st.number_input("Loan Tenure (Years)", 
                                      min_value=1, 
                                      max_value=40, 
                                      value=int(tenure_years_default))
    
    # Calculate button
    if st.button("Calculate EMI", use_container_width=True):
        # Calculate EMI
        emi = calculate_emi(loan_amount, interest_rate, tenure_years)
        
        # Display results
        st.subheader("EMI Calculation Results")
        
        # Show EMI amount
        st.metric("Monthly EMI", f"₹{emi:.2f}")
        
        # Display additional information
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_payment = emi * tenure_years * 12
            st.metric("Total Payment", f"₹{total_payment:.2f}")
        
        with col2:
            total_interest = total_payment - loan_amount
            st.metric("Total Interest", f"₹{total_interest:.2f}")
        
        with col3:
            interest_percentage = (total_interest / loan_amount) * 100
            st.metric("Interest %", f"{interest_percentage:.2f}%")
        
        # Show amortization schedule
        st.subheader("Loan Amortization Schedule")
        
        # Get amortization schedule
        schedule = get_amortization_schedule(loan_amount, interest_rate, tenure_years)
        
        # Allow user to select how many rows to view
        num_rows = st.slider("Select number of months to view", 
                           min_value=12, 
                           max_value=min(tenure_years * 12, 120), 
                           value=24,
                           step=12)
        
        # Display the schedule
        st.dataframe(schedule.head(num_rows).style.format({
            'EMI': '₹{:.2f}',
            'Principal Payment': '₹{:.2f}',
            'Interest Payment': '₹{:.2f}',
            'Remaining Principal': '₹{:.2f}'
        }))
        
        # Visualization of payment breakdown
        st.subheader("Payment Breakdown")
        
        # Create a pie chart of principal vs interest
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pie([loan_amount, total_interest], 
              labels=['Principal', 'Interest'],
              autopct='%1.1f%%',
              startangle=90,
              colors=['#ff9999','#66b3ff'])
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        
        st.pyplot(fig)

def visualization_page(df, model):
    """Data visualization page"""
    st.markdown("<h2 class='sub-header'>📊 Loan Data Visualization</h2>", unsafe_allow_html=True)
    
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["Feature Importance", "Approval Rates", "Applicant Demographics"])
    
    with tab1:
        st.subheader("Feature Importance in Loan Approval")
        
        # Get feature importance
        feature_importance = model.get_feature_importance()
        
        if feature_importance:
            # Sort feature importance
            sorted_features = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
            
            # Create a horizontal bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.barh(list(sorted_features.keys()), list(sorted_features.values()), color='skyblue')
            
            # Add labels
            ax.set_xlabel('Importance')
            ax.set_title('Feature Importance in Loan Approval Decision')
            
            # Add values on bars
            for bar in bars:
                width = bar.get_width()
                ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.4f}', 
                       ha='left', va='center')
            
            st.pyplot(fig)
            
            # Description
            st.markdown("""
            ### Understanding Feature Importance
            
            This chart shows how each factor contributes to the loan approval decision:
            
            - Higher values indicate stronger influence on the prediction
            - Credit history, income, and loan amount are typically strong factors
            - The model uses all these factors combined to make predictions
            """)
        else:
            st.warning("Feature importance data is not available.")
    
    with tab2:
        st.subheader("Loan Approval Rates by Categories")
        
        # Create a copy with Y/N instead of 1/0 for better visualization
        viz_df = df.copy()
        
        # Create approval rate visualizations
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        axs = axs.flatten()
        
        # Define the categorical variables to analyze
        categorical_vars = ['Gender', 'Married', 'Education', 'Self_Employed', 'Credit_History', 'Property_Area']
        
        for i, var in enumerate(categorical_vars):
            # Calculate approval rate by category
            approval_by_category = viz_df.groupby(var)['Loan_Status'].apply(
                lambda x: (x == 'Y').mean()).reset_index()
            
            # Plot
            sns.barplot(x=var, y='Loan_Status', data=approval_by_category, ax=axs[i], color='skyblue')
            axs[i].set_title(f'Approval Rate by {var}')
            axs[i].set_ylabel('Approval Rate')
            axs[i].set_ylim(0, 1)
            
            # Add values on top of bars
            for p in axs[i].patches:
                axs[i].annotate(f'{p.get_height():.2f}', 
                               (p.get_x() + p.get_width() / 2., p.get_height()), 
                               ha='center', va='bottom')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Description
        st.markdown("""
        ### Approval Rate Analysis
        
        These charts show how loan approval rates vary by different categories:
        
        - **Credit History**: Applicants with good credit history (1) have significantly higher approval rates
        - **Property Area**: Location of property affects approval chances
        - **Gender and Marital Status**: These may influence approval decisions
        """)
    
    with tab3:
        st.subheader("Applicant Demographics")
        
        # Create visualizations for applicant demographics
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        axs = axs.flatten()
        
        # Income distribution
        sns.histplot(df['ApplicantIncome'], bins=30, kde=True, ax=axs[0], color='skyblue')
        axs[0].set_title('Applicant Income Distribution')
        axs[0].set_xlabel('Income')
        
        # Loan amount distribution
        sns.histplot(df['LoanAmount'].dropna(), bins=30, kde=True, ax=axs[1], color='lightgreen')
        axs[1].set_title('Loan Amount Distribution')
        axs[1].set_xlabel('Loan Amount (thousands)')
        
        # Loan term distribution
        sns.countplot(x='Loan_Amount_Term', data=df, ax=axs[2], color='salmon')
        axs[2].set_title('Loan Term Distribution')
        axs[2].set_xlabel('Loan Term (months)')
        
        # Dependency distribution
        sns.countplot(x='Dependents', data=df, ax=axs[3], color='mediumpurple')
        axs[3].set_title('Dependents Distribution')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Additional visualization: Income vs. Loan Amount by education
        st.subheader("Income vs. Loan Amount by Education")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.scatterplot(x='ApplicantIncome', y='LoanAmount', 
                       hue='Education', size='Loan_Status',
                       sizes={'Y': 100, 'N': 50},
                       data=df.dropna(subset=['LoanAmount']), ax=ax)
        
        ax.set_title('Income vs. Loan Amount by Education')
        ax.set_xlabel('Applicant Income')
        ax.set_ylabel('Loan Amount (thousands)')
        
        st.pyplot(fig)

def history_page():
    """Page to display prediction history"""
    st.markdown("<h2 class='sub-header'>📝 Prediction History</h2>", unsafe_allow_html=True)
    
    # Get prediction history
    history = get_prediction_history()
    
    if not history:
        st.info("No prediction history available. Make some predictions to see them here.")
        return
    
    # Display the number of predictions
    st.write(f"Total predictions made: {len(history)}")
    
    # Display history in reverse chronological order
    for i, prediction in enumerate(reversed(history)):
        with st.expander(f"Prediction {len(history) - i}: {prediction['timestamp']}"):
            # Create columns for input data and results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Input Data")
                input_data = prediction['input_data']
                
                # Format input data
                formatted_input = pd.DataFrame({
                    'Parameter': list(input_data.keys()),
                    'Value': list(input_data.values())
                })
                
                st.dataframe(formatted_input)
            
            with col2:
                st.subheader("Prediction Results")
                
                # Display prediction result
                result = "APPROVED" if prediction['prediction'] == 1 else "REJECTED"
                probability = prediction['probability']
                
                # Use success/error boxes based on result
                if result == "APPROVED":
                    st.success(f"Loan {result}")
                else:
                    st.error(f"Loan {result}")
                
                # Show probability
                st.write(f"Approval Probability: {probability:.2%}")
                st.progress(probability)
    
    # Option to clear history
    if st.button("Clear Prediction History", use_container_width=True):
        st.session_state['prediction_history'] = []
        st.success("Prediction history cleared!")
        time.sleep(1)
        st.rerun()

def data_explorer_page(df):
    """Page to explore and filter the dataset"""
    st.markdown("<h2 class='sub-header'>🔍 Loan Data Explorer</h2>", unsafe_allow_html=True)
    
    # Dataset Overview
    st.subheader("Dataset Overview")
    st.write(f"Total Records: {len(df)}")
    
    # Show first few rows
    st.dataframe(df.head())
    
    # Column information
    st.write("Dataset Columns:", df.columns.tolist())
    
    # Data Filtering Section
    st.subheader("Filter Dataset")
    
    # Create filter options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Filter by gender
        gender_filter = st.multiselect("Gender", ['Male', 'Female', None], default=['Male', 'Female', None])
    
    with col2:
        # Filter by loan status
        loan_status_filter = st.multiselect("Loan Status", ['Y', 'N'], default=['Y', 'N'])
    
    with col3:
        # Filter by education
        education_filter = st.multiselect("Education", ['Graduate', 'Not Graduate'], default=['Graduate', 'Not Graduate'])
    
    # Income range filter
    income_range = st.slider("Applicant Income Range", 
                          float(df['ApplicantIncome'].min()), 
                          float(df['ApplicantIncome'].max()), 
                          [float(df['ApplicantIncome'].min()), float(df['ApplicantIncome'].max())])
    
    # Apply filters
    filtered_df = df.copy()
    
    # Apply gender filter
    if None in gender_filter:
        filtered_df = filtered_df[filtered_df['Gender'].isin(gender_filter) | filtered_df['Gender'].isna()]
    else:
        filtered_df = filtered_df[filtered_df['Gender'].isin(gender_filter)]
    
    # Apply loan status filter
    filtered_df = filtered_df[filtered_df['Loan_Status'].isin(loan_status_filter)]
    
    # Apply education filter
    filtered_df = filtered_df[filtered_df['Education'].isin(education_filter)]
    
    # Apply income range filter
    filtered_df = filtered_df[(filtered_df['ApplicantIncome'] >= income_range[0]) & 
                            (filtered_df['ApplicantIncome'] <= income_range[1])]
    
    # Display filtered data
    st.subheader("Filtered Data")
    st.write(f"Showing {len(filtered_df)} of {len(df)} records")
    st.dataframe(filtered_df)
    
    # Data Analysis Section
    st.subheader("Data Analysis")
    
    # Descriptive statistics
    if st.checkbox("Show Descriptive Statistics"):
        st.write(filtered_df.describe())
    
    # Correlation matrix
    if st.checkbox("Show Correlation Matrix"):
        # Select only numeric columns
        numeric_df = filtered_df.select_dtypes(include=['number'])
        
        # Calculate correlation
        corr = numeric_df.corr()
        
        # Plot correlation matrix
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        ax.set_title('Correlation Matrix')
        st.pyplot(fig)
    
    # Download filtered data
    st.download_button(
        label="Download Filtered Data as CSV",
        data=filtered_df.to_csv(index=False).encode('utf-8'),
        file_name='filtered_loan_data.csv',
        mime='text/csv',
    )

if __name__ == "__main__":
    main()

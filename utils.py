import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import json
import datetime
from io import BytesIO
import base64

def calculate_emi(principal, interest_rate, tenure_years):
    """
    Calculate EMI using the formula: EMI = [P x R x (1+R)^N]/[(1+R)^N-1]
    
    Parameters:
    principal (float): Loan amount
    interest_rate (float): Annual interest rate (in percentage)
    tenure_years (int): Loan tenure in years
    
    Returns:
    float: Monthly EMI amount
    """
    # Convert interest rate from annual to monthly
    monthly_rate = interest_rate / (12 * 100)
    
    # Convert tenure from years to months
    tenure_months = tenure_years * 12
    
    # Calculate EMI
    if monthly_rate == 0:
        return principal / tenure_months
    
    # Use the EMI formula
    emi = (principal * monthly_rate * (1 + monthly_rate) ** tenure_months) / ((1 + monthly_rate) ** tenure_months - 1)
    
    return emi

def get_amortization_schedule(principal, interest_rate, tenure_years):
    """Generate loan amortization schedule"""
    monthly_rate = interest_rate / (12 * 100)
    tenure_months = tenure_years * 12
    emi = calculate_emi(principal, interest_rate, tenure_years)
    
    # Initialize variables
    remaining_principal = principal
    schedule = []
    
    for month in range(1, int(tenure_months) + 1):
        # Calculate interest for the month
        interest_payment = remaining_principal * monthly_rate
        
        # Calculate principal payment for the month
        principal_payment = emi - interest_payment
        
        # Update remaining principal
        remaining_principal -= principal_payment
        
        if remaining_principal < 0:
            remaining_principal = 0
            
        # Add to schedule
        schedule.append({
            'Month': month,
            'EMI': emi,
            'Principal Payment': principal_payment,
            'Interest Payment': interest_payment,
            'Remaining Principal': remaining_principal
        })
        
    return pd.DataFrame(schedule)

def save_prediction_history(prediction_data):
    """Save prediction history to session state"""
    if 'prediction_history' not in st.session_state:
        st.session_state['prediction_history'] = []
    
    # Add timestamp to prediction data
    prediction_data['timestamp'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Add to history
    st.session_state['prediction_history'].append(prediction_data)

def get_prediction_history():
    """Get prediction history from session state"""
    if 'prediction_history' not in st.session_state:
        st.session_state['prediction_history'] = []
    
    return st.session_state['prediction_history']

def get_dataset_summary(df):
    """Get summary statistics of the dataset"""
    # Basic summary
    summary = {
        'total_records': len(df),
        'approval_rate': df['Loan_Status'].value_counts(normalize=True).get('Y', 0),
        'avg_loan_amount': df['LoanAmount'].mean(),
        'avg_applicant_income': df['ApplicantIncome'].mean(),
        'avg_loan_term': df['Loan_Amount_Term'].mean() / 12  # Convert to years
    }
    
    return summary

def plot_to_base64(fig):
    """Convert a matplotlib figure to base64 string for HTML embedding"""
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    buf.close()
    return img_str

def plot_feature_importance(feature_importance):
    """Plot feature importance of the model"""
    if not feature_importance:
        return None
    
    # Sort feature importance
    sorted_features = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
    
    # Create figure
    fig, ax = plt.figure(figsize=(10, 6)), plt.gca()
    bars = ax.barh(list(sorted_features.keys()), list(sorted_features.values()))
    
    # Add labels and title
    ax.set_xlabel('Importance')
    ax.set_title('Feature Importance')
    
    # Add values on the bars
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.4f}', 
                ha='left', va='center')
    
    plt.tight_layout()
    return fig

def plot_approval_distribution(df):
    """Plot loan approval distribution by categorical variables"""
    # Create a figure with subplots
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    axs = axs.flatten()
    
    categorical_vars = ['Gender', 'Married', 'Education', 'Self_Employed', 'Credit_History', 'Property_Area']
    
    for i, var in enumerate(categorical_vars):
        # Create a pivot table
        pivot = pd.pivot_table(df, values='Loan_Status', 
                               index=var, 
                               aggfunc=lambda x: (x == 'Y').mean())
        
        # Plot
        pivot.plot(kind='bar', ax=axs[i], color='skyblue')
        axs[i].set_title(f'Approval Rate by {var}')
        axs[i].set_ylabel('Approval Rate')
        axs[i].set_ylim(0, 1)
        
        # Add values on top of the bars
        for p in axs[i].patches:
            axs[i].annotate(f'{p.get_height():.2f}', 
                           (p.get_x() + p.get_width() / 2., p.get_height()), 
                           ha='center', va='bottom')
    
    plt.tight_layout()
    return fig

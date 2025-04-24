import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

class LoanPredictionModel:
    def __init__(self):
        """Initialize the loan prediction model"""
        self.model = GradientBoostingClassifier(random_state=42)
        self.label_encoders = {}
        self.feature_importance = None
        self.accuracy = None
        self.X_columns = None
    
    def load_and_preprocess_data(self, data_path):
        """Load and preprocess the data"""
        # Load the dataset
        df = pd.read_csv(data_path)
        
        # Drop rows with missing values
        df = df.dropna()
        
        # Encode categorical variables
        for col in ['Gender', 'Married', 'Dependents', 'Education',
                    'Self_Employed', 'Property_Area']:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.label_encoders[col] = le
        
        # Map the target variable
        df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})
        
        return df
    
    def train(self, data_path):
        """Train the model on the dataset"""
        # Load and preprocess data
        df = self.load_and_preprocess_data(data_path)
        
        # Split into features and target
        X = df.drop(['Loan_ID', 'Loan_Status'], axis=1)
        self.X_columns = X.columns
        y = df['Loan_Status']
        
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = self.model.predict(X_test)
        self.accuracy = accuracy_score(y_test, y_pred)
        self.report = classification_report(y_test, y_pred)
        
        # Get feature importances
        self.feature_importance = dict(zip(X.columns, self.model.feature_importances_))
        
        return self.accuracy, self.report, self.feature_importance
    
    def preprocess_input(self, input_data):
        """Preprocess user input data for prediction"""
        # Create a DataFrame from input data
        input_dict = {
            'Gender': self.label_encoders['Gender'].transform([input_data['Gender']])[0],
            'Married': self.label_encoders['Married'].transform([input_data['Married']])[0],
            'Dependents': self.label_encoders['Dependents'].transform([input_data['Dependents']])[0],
            'Education': self.label_encoders['Education'].transform([input_data['Education']])[0],
            'Self_Employed': self.label_encoders['Self_Employed'].transform([input_data['Self_Employed']])[0],
            'ApplicantIncome': input_data['ApplicantIncome'],
            'CoapplicantIncome': input_data['CoapplicantIncome'],
            'LoanAmount': input_data['LoanAmount'],
            'Loan_Amount_Term': input_data['Loan_Amount_Term'],
            'Credit_History': input_data['Credit_History'],
            'Property_Area': self.label_encoders['Property_Area'].transform([input_data['Property_Area']])[0],
        }
        
        input_df = pd.DataFrame([input_dict])
        return input_df
    
    def predict(self, input_data):
        """Make a prediction based on user input"""
        # Preprocess the input data
        input_df = self.preprocess_input(input_data)
        
        # Make prediction
        probability = self.model.predict_proba(input_df)[0][1]
        prediction = 1 if probability > 0.5 else 0
        
        return prediction, probability
    
    def get_feature_importance(self):
        """Get the feature importance of the model"""
        return self.feature_importance

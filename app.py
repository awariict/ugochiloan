"""
🏦 MICROFINANCE LOAN DEFAULT PREDICTION SYSTEM
Enhanced with Auto-Approval, Smart Loan Offers & Advanced Forecasting
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from werkzeug.security import generate_password_hash, check_password_hash
import json
import plotly.express as px
import plotly.graph_objects as go
import warnings
import os
from pymongo import MongoClient
from bson.objectid import ObjectId
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression

warnings.filterwarnings('ignore')

# ========================================
# 1. PAGE CONFIG
# ========================================

st.set_page_config(page_title="Microfinance System", layout="wide")

# ===== UI WITH BLUE SIDEBAR =====

st.markdown("""<style>
.stApp { background: linear-gradient(to right, #007BFF, #FFC107, #FF0000); }
section[data-testid="stSidebar"] { background: linear-gradient(to bottom, #0056b3, #003d82)!important; }
.sidebar-text { color: white !important; }
.card { background: white; padding: 12px; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.08); margin-bottom: 8px; }
.alert-warning { background-color: #fff3cd; border-left: 4px solid #ffc107; padding: 12px; margin: 10px 0; }
.alert-danger { background-color: #f8d7da; border-left: 4px solid #dc3545; padding: 12px; margin: 10px 0; }
.alert-info { background-color: #d1ecf1; border-left: 4px solid #17a2b8; padding: 12px; margin: 10px 0; }
.alert-success { background-color: #d4edda; border-left: 4px solid #28a745; padding: 12px; margin: 10px 0; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] p, [data-testid="stSidebar"] span { color: white !important; }
</style>""", unsafe_allow_html=True)

# ========================================
# 2. MONGODB CONNECTION
# ========================================
MONGODB_URI = "mongodb+srv://euawari_db_user:6SnKvQvXXzrGeypA@cluster0.fkkzcvz.mongodb.net/microfinance_db?retryWrites=true&w=majority"

@st.cache_resource
def init_mongodb():
    """Initialize MongoDB connection"""
    client = MongoClient(MONGODB_URI, maxPoolSize=50, minPoolSize=10)
    return client["microfinance_db"]

db = init_mongodb()

# ========================================
# 3. SESSION STATE
# ========================================
if "user" not in st.session_state:
    st.session_state.user = None
if "models" not in st.session_state:
    st.session_state.models = None

# ========================================
# 4. UTILITY FUNCTIONS
# ========================================
def safe_num(val, default=0, is_float=False):
    """Convert value to number safely"""
    try:
        if pd.isna(val):
            return float(default) if is_float else int(default)
        return float(val) if is_float else int(float(val))
    except:
        return float(default) if is_float else int(default)

def clean_df(df):
    """Clean dataframe - fill numeric nulls"""
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        median = df[col].median()
        df[col].fillna(median if pd.notna(median) else 0, inplace=True)
    return df

# ========================================
# 5. AUTHENTICATION
# ========================================
def register(username, password, role):
    """Register new user"""
    try:
        if db['users'].find_one({"username": username}):
            return "exists"
        
        db['users'].insert_one({
            "username": username,
            "password": generate_password_hash(password),
            "role": role,
            "status": "pending",
            "created_at": datetime.now()
        })
        return "success"
    except Exception as e:
        st.error(f"Error: {e}")
        return "error"

def login(username, password):
    """Login user"""
    try:
        user = db['users'].find_one({"username": username})
        if not user:
            return None
        if user["status"] != "approved":
            return "PENDING"
        if check_password_hash(user["password"], password):
            return {"_id": str(user["_id"]), "username": username, "role": user["role"]}
        return None
    except:
        return None

# ========================================
# 6. BORROWER MANAGEMENT
# ========================================
def add_borrower(name, age, income, repayment=80, prev_loans=0, defaults=0, txn_freq=5):
    """Add single borrower"""
    try:
        db['borrowers'].insert_one({
            "name": name,
            "age": safe_num(age),
            "income": safe_num(income, is_float=True),
            "repayment_history": safe_num(repayment, is_float=True),
            "previous_loans": safe_num(prev_loans),
            "defaults": safe_num(defaults),
            "transaction_freq": safe_num(txn_freq, is_float=True),
            "created_at": datetime.now()
        })
        return True
    except:
        return False

def get_borrowers():
    """Get all borrowers"""
    try:
        data = list(db['borrowers'].find())
        if data:
            df = pd.DataFrame(data)
            df['id'] = df['_id'].astype(str)
            df = clean_df(df)
            return df
        return pd.DataFrame()
    except:
        return pd.DataFrame()

def bulk_upload_borrowers(file_df):
    """Bulk upload borrowers from file"""
    count = 0
    errors = []
    
    for idx, row in file_df.iterrows():
        try:
            name = str(row.get('name', ''))
            if not name:
                errors.append(f"Row {idx+1}: Missing name")
                continue
            
            if add_borrower(
                name=name,
                age=row.get('age', 25),
                income=row.get('income', 10000),
                repayment=row.get('repayment_history', 80),
                prev_loans=row.get('previous_loans', 0),
                defaults=row.get('defaults', 0),
                txn_freq=row.get('transaction_freq', 5)
            ):
                count += 1
            else:
                errors.append(f"Row {idx+1}: Failed to insert")
        except Exception as e:
            errors.append(f"Row {idx+1}: {str(e)}")
    
    return count, errors

# ========================================
# 7. LOAN MANAGEMENT
# ========================================
def calculate_loan_amount(balance, txn_history_count, defaults=0, repayment_rate=80):
    """Calculate smart loan offer based on customer profile"""
    try:
        # Base on account balance (3-5x multiplier)
        balance_factor = balance * 3.5 if balance > 0 else 0
        
        # Transaction activity bonus (active customers get more)
        activity_bonus = min(txn_history_count * 1000, balance * 1.5)
        
        # Default penalty
        default_penalty = 1.0 if defaults == 0 else max(0.3, 1 - (defaults * 0.2))
        
        # Repayment history factor
        repayment_factor = repayment_rate / 100
        
        base_amount = (balance_factor + activity_bonus) * repayment_factor * default_penalty
        
        # Minimum 5000, maximum 5x account balance
        loan_amount = max(5000, min(base_amount, balance * 5))
        
        return loan_amount
    except:
        return 5000

def get_loan_decision(risk_score, balance):
    """Get loan decision based on risk"""
    if risk_score >= 60:
        return "REJECT", f"High risk score: {risk_score:.2f}%"
    if risk_score >= 40:
        return "REVIEW", f"Medium risk score: {risk_score:.2f}%"
    return "APPROVE", f"Low risk score: {risk_score:.2f}%"

def calculate_repayment_schedule(loan_amount, duration_months):
    """Calculate detailed repayment schedule"""
    try:
        # Simple interest calculation
        interest_rate = 0.05  # 5% annual
        total_interest = loan_amount * interest_rate * (duration_months / 12)
        total_amount = loan_amount + total_interest
        
        # Daily, weekly, monthly, quarterly payments
        daily_payment = total_amount / (duration_months * 30)
        weekly_payment = daily_payment * 7
        monthly_payment = total_amount / duration_months
        quarterly_payment = total_amount / (duration_months / 3)
        
        return {
            "total_amount": total_amount,
            "interest": total_interest,
            "daily": daily_payment,
            "weekly": weekly_payment,
            "monthly": monthly_payment,
            "quarterly": quarterly_payment
        }
    except:
        return None

def predict_eligibility_timeline(risk_score, repayment_history):
    """Predict when rejected customer becomes eligible"""
    try:
        if risk_score < 60:
            return None  # Already eligible
        
        # Estimate months to become eligible (reduce risk to 60%)
        reduction_needed = risk_score - 60
        months_to_eligible = int(np.ceil(reduction_needed / 10))  # 10% reduction per month with good behavior
        
        eligible_date = datetime.now() + timedelta(days=months_to_eligible * 30)
        
        # Suggest max loan at that time
        suggested_loan = 15000 * (repayment_history / 100) if repayment_history > 0 else 10000
        
        return {
            "months_to_eligible": months_to_eligible,
            "eligible_date": eligible_date,
            "suggested_loan_amount": suggested_loan,
            "required_improvements": [
                "Make all payments on time",
                "Increase transaction frequency",
                "Maintain consistent income",
                "Avoid any defaults"
            ]
        }
    except:
        return None

def create_loan(bid, amount, duration, risk, model, decision, reason, borrowed_amount=0):
    """Create loan record"""
    try:
        status_map = {'APPROVE': 'approved', 'REJECT': 'rejected', 'REVIEW': 'pending_review'}
        db['loans'].insert_one({
            "borrower_id": bid,
            "amount": safe_num(amount, is_float=True),
            "duration": safe_num(duration),
            "risk_score": safe_num(risk, is_float=True),
            "model_name": model,
            "status": status_map.get(decision, 'pending_review'),
            "decision_reason": reason,
            "borrowed_amount": safe_num(borrowed_amount, is_float=True),
            "actual_default": 0,
            "created_at": datetime.now(),
            "approved_at": datetime.now() if decision == "APPROVE" else None
        })
        return True
    except:
        return False

def get_loans():
    """Get all loans"""
    try:
        data = list(db['loans'].find())
        if data:
            df = pd.DataFrame(data)
            df['id'] = df['_id'].astype(str)
            return df
        return pd.DataFrame()
    except:
        return pd.DataFrame()

def get_loan_decisions():
    """Get loans with borrower info"""
    try:
        loans = list(db['loans'].find().sort("created_at", -1))
        results = []
        
        for loan in loans:
            try:
                bid = loan.get("borrower_id")
                borrower = db['borrowers'].find_one(
                    {"_id": ObjectId(bid)} if isinstance(bid, str) and len(str(bid)) == 24 
                    else {"_id": bid}
                )
                
                if borrower:
                    results.append({
                        "loan_id": str(loan["_id"]),
                        "borrower_name": borrower.get("name", ""),
                        "income": borrower.get("income", 0),
                        "age": borrower.get("age", 0),
                        "repayment": borrower.get("repayment_history", 0),
                        "prev_loans": borrower.get("previous_loans", 0),
                        "defaults": borrower.get("defaults", 0),
                        "amount": loan.get("amount", 0),
                        "duration": loan.get("duration", 0),
                        "risk": loan.get("risk_score", 0),
                        "model": loan.get("model_name", ""),
                        "status": loan.get("status", ""),
                        "reason": loan.get("decision_reason", ""),
                        "created": loan.get("created_at")
                    })
            except:
                pass
        
        return pd.DataFrame(results) if results else pd.DataFrame()
    except:
        return pd.DataFrame()

# ========================================
# 8. MODEL TRAINING WITH AUTO-APPROVAL
# ========================================
def train_models(borrowers_df):
    """Train ML models with cross-validation"""
    if len(borrowers_df) < 20:
        return None
    
    try:
        borrowers_df["default_flag"] = (borrowers_df["defaults"] > 0).astype(int)
        features = ["income", "age", "repayment_history", "previous_loans", "transaction_freq"]
        
        X = borrowers_df[features].copy()
        y = borrowers_df["default_flag"].copy()
        X = X.fillna(X.mean())
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        try:
            smote = SMOTE(random_state=42, k_neighbors=min(3, max(1, (y_train == 1).sum() - 1)))
            X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
        except:
            X_train_smote, y_train_smote = X_train, y_train
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_smote)
        X_test_scaled = scaler.transform(X_test)
        
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        }
        
        results = {}
        for model_name, model in models.items():
            if model_name == "Logistic Regression":
                model.fit(X_train_scaled, y_train_smote)
                pred = model.predict(X_test_scaled)
                proba = model.predict_proba(X_test_scaled)[:, 1]
                cv_scores = cross_val_score(model, X_train_scaled, y_train_smote, cv=5, scoring='roc_auc')
            else:
                model.fit(X_train_smote, y_train_smote)
                pred = model.predict(X_test)
                proba = model.predict_proba(X_test)[:, 1]
                cv_scores = cross_val_score(model, X_train_smote, y_train_smote, cv=5, scoring='roc_auc')
            
            fpr, tpr, _ = roc_curve(y_test, proba)
            
            results[model_name] = {
                "Accuracy": accuracy_score(y_test, pred),
                "Precision": precision_score(y_test, pred, zero_division=0),
                "Recall": recall_score(y_test, pred, zero_division=0),
                "F1": f1_score(y_test, pred, zero_division=0),
                "AUC": roc_auc_score(y_test, proba),
                "CV_Mean": cv_scores.mean(),
                "CV_Std": cv_scores.std(),
                "model": model,
                "scaler": scaler if model_name == "Logistic Regression" else None,
                "y_test": y_test,
                "y_pred": pred,
                "y_prob": proba,
                "fpr": fpr,
                "tpr": tpr,
                "features": features,
                "X_test": X_test if model_name != "Logistic Regression" else X_test_scaled
            }
        
        return results
    except Exception as e:
        st.error(f"Training error: {e}")
        return None

def predict_risk(model, scaler, data, model_name):
    """Predict default risk"""
    try:
        if model_name == "Logistic Regression" and scaler:
            data_scaled = scaler.transform([data])
            risk = model.predict_proba(data_scaled)[0][1] * 100
        else:
            risk = model.predict_proba([data])[0][1] * 100
        return risk
    except:
        return 50

def auto_process_loan_applications():
    """Auto-process pending customer loan applications"""
    try:
        if not st.session_state.models:
            return 0
        
        pending_apps = list(db['loan_applications'].find({"status": "pending"}))
        processed = 0
        
        for app in pending_apps:
            try:
                borrower_data = db['borrowers'].find_one({"name": app.get("username")})
                
                if not borrower_data:
                    income = app.get("income", 50000)
                    age = app.get("age", 30)
                    repayment = app.get("repayment_history", 80)
                    prev_loans = app.get("previous_loans", 0)
                    defaults = app.get("defaults", 0)
                    txn_freq = app.get("transaction_freq", 5)
                else:
                    income = borrower_data.get("income", 50000)
                    age = borrower_data.get("age", 30)
                    repayment = borrower_data.get("repayment_history", 80)
                    prev_loans = borrower_data.get("previous_loans", 0)
                    defaults = borrower_data.get("defaults", 0)
                    txn_freq = borrower_data.get("transaction_freq", 5)
                
                models = st.session_state.models
                best_model_name = max(models.items(), key=lambda x: x[1]["AUC"])[0]
                best_model_info = models[best_model_name]
                
                data = [income, age, repayment, prev_loans, txn_freq]
                risk = predict_risk(
                    best_model_info['model'],
                    best_model_info['scaler'],
                    data,
                    best_model_name
                )
                
                decision, reason = get_loan_decision(risk, app.get("offered_amount", 0))
                
                db['loan_applications'].update_one(
                    {"_id": app["_id"]},
                    {
                        "$set": {
                            "status": decision.lower() if decision != "REVIEW" else "pending",
                            "decision_reason": reason,
                            "risk_score": risk,
                            "model_used": best_model_name,
                            "processed_at": datetime.now()
                        }
                    }
                )
                
                processed += 1
            except:
                pass
        
        return processed
    except:
        return 0

# ========================================
# 9. EDA FUNCTIONS
# ========================================
def generate_eda_charts(df):
    """Generate EDA visualizations"""
    charts = {}
    
    try:
        fig_scatter = px.scatter(df, x="income", y="defaults", 
                                 color="repayment_history", size="age",
                                 title="Income vs Defaults (colored by Repayment %)",
                                 labels={"income": "Income ($)", "defaults": "Number of Defaults"})
        fig_scatter.update_layout(template="plotly_dark")
        charts["scatter_income_defaults"] = fig_scatter
    except:
        pass
    
    try:
        fig_scatter2 = px.scatter(df, x="age", y="repayment_history",
                                  color="defaults", size="income",
                                  title="Age vs Repayment History (colored by Defaults)",
                                  labels={"age": "Age (years)", "repayment_history": "Repayment %"})
        fig_scatter2.update_layout(template="plotly_dark")
        charts["scatter_age_repayment"] = fig_scatter2
    except:
        pass
    
    try:
        fig_dist_income = px.histogram(df, x="income", nbins=30, 
                                       title="Income Distribution",
                                       labels={"income": "Annual Income ($)"},
                                       color_discrete_sequence=["#3498db"])
        fig_dist_income.update_layout(template="plotly_dark")
        charts["dist_income"] = fig_dist_income
    except:
        pass
    
    try:
        fig_dist_age = px.histogram(df, x="age", nbins=20,
                                    title="Age Distribution",
                                    labels={"age": "Age (years)"},
                                    color_discrete_sequence=["#2ecc71"])
        fig_dist_age.update_layout(template="plotly_dark")
        charts["dist_age"] = fig_dist_age
    except:
        pass
    
    try:
        fig_dist_repayment = px.histogram(df, x="repayment_history", nbins=20,
                                          title="Repayment History Distribution",
                                          labels={"repayment_history": "Repayment %"},
                                          color_discrete_sequence=["#e74c3c"])
        fig_dist_repayment.update_layout(template="plotly_dark")
        charts["dist_repayment"] = fig_dist_repayment
    except:
        pass
    
    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numeric_cols].corr()
        
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0
        ))
        fig_corr.update_layout(title="Feature Correlation Heatmap", template="plotly_dark")
        charts["correlation"] = fig_corr
    except:
        pass
    
    try:
        fig_box = px.box(df, y=["income", "repayment_history", "transaction_freq"],
                         title="Statistical Distribution of Key Features",
                         labels={"variable": "Feature", "value": "Value"})
        fig_box.update_layout(template="plotly_dark")
        charts["box_plot"] = fig_box
    except:
        pass
    
    return charts

# ========================================
# 10. ROBUST FORECASTING FUNCTIONS
# ========================================
def forecast_loan_amounts(loans_df, periods=30):
    """Forecast average loan amounts - Works with minimal data"""
    try:
        if len(loans_df) == 0:
            return None
        
        loans_df = loans_df.copy()
        loans_df['created_at'] = pd.to_datetime(loans_df['created_at'])
        loans_df = loans_df.sort_values('created_at')
        
        # If we have at least one record, use simple forecasting
        if len(loans_df) == 1:
            avg_amount = loans_df['amount'].mean()
            dates = pd.date_range(datetime.now(), periods=periods, freq='D')
            forecast_values = [avg_amount] * periods
            
            forecast_df = pd.DataFrame({
                'forecast': forecast_values
            }, index=dates)
            
            historical_df = pd.DataFrame({
                'amount': [avg_amount]
            }, index=[loans_df['created_at'].iloc[0]])
            
            return forecast_df, historical_df['amount'], None
        
        # Group by day and get average
        loans_df.set_index('created_at', inplace=True)
        daily_avg = loans_df['amount'].resample('D').mean().fillna(method='ffill')
        
        if len(daily_avg) < 2:
            daily_avg = daily_avg.reindex(pd.date_range(daily_avg.index[0], periods=max(2, len(daily_avg)), freq='D'), method='ffill')
        
        try:
            # Try ARIMA if we have enough points
            if len(daily_avg) >= 4:
                model = ARIMA(daily_avg, order=(1, 0, 0), suppress_warnings=True)
                fitted = model.fit()
                forecast = fitted.get_forecast(steps=periods, alpha=0.05)
                forecast_df = forecast.conf_int(alpha=0.05).copy()
                forecast_df.columns = ['lower', 'upper']
                forecast_df['forecast'] = forecast.predicted_mean
            else:
                raise Exception("Not enough data for ARIMA")
        except:
            # Fallback to linear regression
            X = np.arange(len(daily_avg)).reshape(-1, 1)
            y = daily_avg.values
            model = LinearRegression()
            model.fit(X, y)
            
            future_X = np.arange(len(daily_avg), len(daily_avg) + periods).reshape(-1, 1)
            forecast_values = model.predict(future_X)
            
            dates = pd.date_range(daily_avg.index[-1] + timedelta(days=1), periods=periods, freq='D')
            forecast_df = pd.DataFrame({
                'forecast': forecast_values,
                'lower': forecast_values * 0.9,
                'upper': forecast_values * 1.1
            }, index=dates)
        
        return forecast_df, daily_avg, model
    except Exception as e:
        st.warning(f"Forecasting error: {str(e)}")
        return None

def forecast_default_risk(loans_df, periods=30):
    """Forecast default risk trends"""
    try:
        if len(loans_df) == 0:
            return None
        
        loans_df = loans_df.copy()
        loans_df['created_at'] = pd.to_datetime(loans_df['created_at'])
        loans_df = loans_df.sort_values('created_at')
        
        if len(loans_df) == 1:
            avg_risk = loans_df['risk_score'].mean()
            dates = pd.date_range(datetime.now(), periods=periods, freq='D')
            forecast_values = [avg_risk] * periods
            
            forecast_df = pd.DataFrame({
                'forecast': forecast_values
            }, index=dates)
            
            historical_df = pd.DataFrame({
                'risk_score': [avg_risk]
            }, index=[loans_df['created_at'].iloc[0]])
            
            return forecast_df, historical_df['risk_score'], None
        
        loans_df.set_index('created_at', inplace=True)
        daily_risk = loans_df['risk_score'].resample('D').mean().fillna(method='ffill')
        
        if len(daily_risk) < 2:
            daily_risk = daily_risk.reindex(pd.date_range(daily_risk.index[0], periods=max(2, len(daily_risk)), freq='D'), method='ffill')
        
        try:
            if len(daily_risk) >= 4:
                model = ARIMA(daily_risk, order=(1, 0, 0), suppress_warnings=True)
                fitted = model.fit()
                forecast = fitted.get_forecast(steps=periods, alpha=0.05)
                forecast_df = forecast.conf_int(alpha=0.05).copy()
                forecast_df.columns = ['lower', 'upper']
                forecast_df['forecast'] = forecast.predicted_mean
            else:
                raise Exception("Not enough data")
        except:
            X = np.arange(len(daily_risk)).reshape(-1, 1)
            y = daily_risk.values
            model = LinearRegression()
            model.fit(X, y)
            
            future_X = np.arange(len(daily_risk), len(daily_risk) + periods).reshape(-1, 1)
            forecast_values = model.predict(future_X)
            
            dates = pd.date_range(daily_risk.index[-1] + timedelta(days=1), periods=periods, freq='D')
            forecast_df = pd.DataFrame({
                'forecast': forecast_values,
                'lower': forecast_values * 0.9,
                'upper': forecast_values * 1.1
            }, index=dates)
        
        return forecast_df, daily_risk, model
    except Exception as e:
        st.warning(f"Risk forecasting error: {str(e)}")
        return None

def forecast_approval_rate(loans_df, periods=30):
    """Forecast approval rate"""
    try:
        if len(loans_df) == 0:
            return None
        
        loans_df = loans_df.copy()
        loans_df['created_at'] = pd.to_datetime(loans_df['created_at'])
        loans_df = loans_df.sort_values('created_at')
        loans_df['approved'] = (loans_df['status'] == 'approved').astype(int)
        
        if len(loans_df) == 1:
            approval_rate = loans_df['approved'].mean() * 100
            dates = pd.date_range(datetime.now(), periods=periods, freq='D')
            forecast_values = [approval_rate] * periods
            
            forecast_df = pd.DataFrame({
                'forecast': forecast_values
            }, index=dates)
            
            historical_df = pd.DataFrame({
                'approval_rate': [approval_rate]
            }, index=[loans_df['created_at'].iloc[0]])
            
            return forecast_df, historical_df['approval_rate'], None
        
        loans_df.set_index('created_at', inplace=True)
        daily_approval = loans_df['approved'].resample('D').mean().fillna(method='ffill') * 100
        
        if len(daily_approval) < 2:
            daily_approval = daily_approval.reindex(pd.date_range(daily_approval.index[0], periods=max(2, len(daily_approval)), freq='D'), method='ffill')
        
        try:
            if len(daily_approval) >= 4:
                model = ARIMA(daily_approval, order=(1, 0, 0), suppress_warnings=True)
                fitted = model.fit()
                forecast = fitted.get_forecast(steps=periods, alpha=0.05)
                forecast_df = forecast.conf_int(alpha=0.05).copy()
                forecast_df.columns = ['lower', 'upper']
                forecast_df['forecast'] = forecast.predicted_mean
            else:
                raise Exception("Not enough data")
        except:
            X = np.arange(len(daily_approval)).reshape(-1, 1)
            y = daily_approval.values
            model = LinearRegression()
            model.fit(X, y)
            
            future_X = np.arange(len(daily_approval), len(daily_approval) + periods).reshape(-1, 1)
            forecast_values = model.predict(future_X)
            
            dates = pd.date_range(daily_approval.index[-1] + timedelta(days=1), periods=periods, freq='D')
            forecast_df = pd.DataFrame({
                'forecast': forecast_values,
                'lower': np.maximum(forecast_values * 0.9, 0),
                'upper': np.minimum(forecast_values * 1.1, 100)
            }, index=dates)
        
        return forecast_df, daily_approval, model
    except Exception as e:
        st.warning(f"Approval rate forecasting error: {str(e)}")
        return None

# ========================================
# 11. BANKING FUNCTIONS FOR NORMAL USERS
# ========================================
def create_account(username, password):
    """Create a bank account for normal user"""
    try:
        user_id = db['users'].find_one({"username": username})["_id"]
        if db['accounts'].find_one({"user_id": user_id}):
            return "exists"
        
        db['accounts'].insert_one({
            "user_id": user_id,
            "username": username,
            "balance": 0,
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        })
        return "success"
    except:
        return "error"

def get_account_balance(user_id):
    """Get account balance"""
    try:
        account = db['accounts'].find_one({"user_id": ObjectId(user_id)})
        return account.get("balance", 0) if account else 0
    except:
        return 0

def get_transaction_count(user_id, days=30):
    """Get transaction count for last N days"""
    try:
        cutoff_date = datetime.now() - timedelta(days=days)
        count = db['transactions'].count_documents({
            "user_id": user_id,
            "timestamp": {"$gte": cutoff_date}
        })
        return count
    except:
        return 0

def deposit_money(user_id, amount):
    """Deposit money to account"""
    try:
        if amount <= 0:
            return False
        
        result = db['accounts'].update_one(
            {"user_id": ObjectId(user_id)},
            {
                "$inc": {"balance": amount},
                "$set": {"updated_at": datetime.now()}
            }
        )
        
        if result.matched_count > 0:
            db['transactions'].insert_one({
                "user_id": user_id,
                "type": "deposit",
                "amount": amount,
                "timestamp": datetime.now(),
                "status": "completed"
            })
            return True
        return False
    except:
        return False

def withdraw_money(user_id, amount):
    """Withdraw money from account"""
    try:
        if amount <= 0:
            return "invalid"
        
        account = db['accounts'].find_one({"user_id": ObjectId(user_id)})
        if not account or account['balance'] < amount:
            return "insufficient"
        
        result = db['accounts'].update_one(
            {"user_id": ObjectId(user_id)},
            {
                "$inc": {"balance": -amount},
                "$set": {"updated_at": datetime.now()}
            }
        )
        
        if result.matched_count > 0:
            db['transactions'].insert_one({
                "user_id": user_id,
                "type": "withdrawal",
                "amount": amount,
                "timestamp": datetime.now(),
                "status": "completed"
            })
            return "success"
        return "error"
    except:
        return "error"

def transfer_money(user_id, recipient_username, amount):
    """Transfer money to another user"""
    try:
        if amount <= 0:
            return "invalid"
        
        sender_account = db['accounts'].find_one({"user_id": ObjectId(user_id)})
        if not sender_account or sender_account['balance'] < amount:
            return "insufficient"
        
        recipient = db['users'].find_one({"username": recipient_username})
        if not recipient:
            return "not_found"
        
        recipient_account = db['accounts'].find_one({"user_id": recipient["_id"]})
        if not recipient_account:
            return "recipient_no_account"
        
        db['accounts'].update_one(
            {"user_id": ObjectId(user_id)},
            {
                "$inc": {"balance": -amount},
                "$set": {"updated_at": datetime.now()}
            }
        )
        
        db['accounts'].update_one(
            {"user_id": recipient["_id"]},
            {
                "$inc": {"balance": amount},
                "$set": {"updated_at": datetime.now()}
            }
        )
        
        db['transactions'].insert_one({
            "user_id": user_id,
            "type": "transfer_out",
            "recipient": recipient_username,
            "amount": amount,
            "timestamp": datetime.now(),
            "status": "completed"
        })
        
        db['transactions'].insert_one({
            "user_id": str(recipient["_id"]),
            "type": "transfer_in",
            "sender": sender_account['username'],
            "amount": amount,
            "timestamp": datetime.now(),
            "status": "completed"
        })
        
        return "success"
    except:
        return "error"

def recharge_card(user_id, amount, card_number):
    """Recharge card (virtual card service)"""
    try:
        if amount <= 0:
            return "invalid"
        
        account = db['accounts'].find_one({"user_id": ObjectId(user_id)})
        if not account or account['balance'] < amount:
            return "insufficient"
        
        db['accounts'].update_one(
            {"user_id": ObjectId(user_id)},
            {
                "$inc": {"balance": -amount},
                "$set": {"updated_at": datetime.now()}
            }
        )
        
        db['transactions'].insert_one({
            "user_id": user_id,
            "type": "card_recharge",
            "amount": amount,
            "card_number": f"****{card_number[-4:]}",
            "timestamp": datetime.now(),
            "status": "completed"
        })
        
        return "success"
    except:
        return "error"

def apply_for_loan_user(user_id, username, offered_amount):
    """User applies for loan with system-calculated offer"""
    try:
        db['loan_applications'].insert_one({
            "user_id": user_id,
            "username": username,
            "offered_amount": offered_amount,
            "accepted_amount": 0,
            "duration": 12,
            "status": "pending",
            "created_at": datetime.now(),
            "decision_reason": "Waiting for model processing"
        })
        return True
    except:
        return False

def get_user_loan_status(user_id):
    """Get user's loan application status"""
    try:
        app = db['loan_applications'].find_one({"user_id": user_id}, sort=[("created_at", -1)])
        if app:
            schedule = None
            days_left = None
            
            if app.get("status") == "approved" and app.get("accepted_amount"):
                schedule = calculate_repayment_schedule(app["accepted_amount"], app.get("duration", 12))
                if app.get("created_at"):
                    end_date = app["created_at"] + timedelta(days=app.get("duration", 12) * 30)
                    days_left = max(0, (end_date - datetime.now()).days)
            
            return {
                "status": app.get("status"),
                "offered_amount": app.get("offered_amount"),
                "accepted_amount": app.get("accepted_amount"),
                "duration": app.get("duration"),
                "reason": app.get("decision_reason"),
                "created_at": app.get("created_at"),
                "days_left": days_left,
                "schedule": schedule,
                "risk_score": app.get("risk_score"),
                "id": str(app["_id"])
            }
        return None
    except:
        return None

def accept_loan_offer(user_id, amount, duration):
    """Accept loan offer and trigger ML evaluation"""
    try:
        app = db['loan_applications'].find_one({"user_id": user_id}, sort=[("created_at", -1)])
        if not app:
            return False
        
        db['loan_applications'].update_one(
            {"_id": app["_id"]},
            {
                "$set": {
                    "accepted_amount": amount,
                    "duration": duration,
                    "offer_accepted_at": datetime.now()
                }
            }
        )
        return True
    except:
        return False

def withdraw_loan_money(user_id, amount):
    """Withdraw approved loan money to account"""
    try:
        app = db['loan_applications'].find_one({"user_id": user_id, "status": "approved"}, sort=[("created_at", -1)])
        if not app:
            return "not_approved"
        
        if app.get("borrowed_amount", 0) + amount > app.get("accepted_amount", 0):
            return "exceeds_limit"
        
        db['accounts'].update_one(
            {"user_id": ObjectId(user_id)},
            {
                "$inc": {"balance": amount},
                "$set": {"updated_at": datetime.now()}
            }
        )
        
        db['loan_applications'].update_one(
            {"_id": app["_id"]},
            {"$inc": {"borrowed_amount": amount}}
        )
        
        db['transactions'].insert_one({
            "user_id": user_id,
            "type": "loan_withdrawal",
            "amount": amount,
            "timestamp": datetime.now(),
            "status": "completed"
        })
        
        return "success"
    except:
        return "error"

def get_user_transactions(user_id, limit=20):
    """Get user transaction history"""
    try:
        transactions = list(db['transactions'].find(
            {"user_id": user_id}
        ).sort("timestamp", -1).limit(limit))
        
        if transactions:
            return pd.DataFrame(transactions)
        return pd.DataFrame()
    except:
        return pd.DataFrame()

# ========================================
# 12. USER MANAGEMENT (ADMIN)
# ========================================
def get_pending_users():
    """Get pending user approvals"""
    try:
        users = list(db['users'].find({"status": "pending"}))
        return pd.DataFrame(users) if users else pd.DataFrame()
    except:
        return pd.DataFrame()

def get_all_users():
    """Get all users with details"""
    try:
        users = list(db['users'].find())
        if users:
            df = pd.DataFrame(users)
            df['_id'] = df['_id'].astype(str)
            return df
        return pd.DataFrame()
    except:
        return pd.DataFrame()

def approve_user(user_id):
    """Approve pending user"""
    try:
        db['users'].update_one({"_id": ObjectId(user_id)}, {"$set": {"status": "approved"}})
        return True
    except:
        return False

def delete_user(user_id):
    """Delete user permanently"""
    try:
        db['users'].delete_one({"_id": ObjectId(user_id)})
        db['accounts'].delete_one({"user_id": ObjectId(user_id)})
        db['transactions'].delete_many({"user_id": user_id})
        return True
    except:
        return False

def delete_all_transactions():
    """Delete all transactions"""
    try:
        db['transactions'].delete_many({})
        return True
    except:
        return False

def delete_all_users_and_transactions():
    """Delete all users and transactions"""
    try:
        db['users'].delete_many({})
        db['accounts'].delete_many({})
        db['transactions'].delete_many({})
        return True
    except:
        return False

def get_all_loan_applications():
    """Get all customer loan applications"""
    try:
        apps = list(db['loan_applications'].find().sort("created_at", -1))
        if apps:
            return pd.DataFrame(apps)
        return pd.DataFrame()
    except:
        return pd.DataFrame()

def update_loan_application_status(app_id, status, reason):
    """Update loan application status manually"""
    try:
        db['loan_applications'].update_one(
            {"_id": ObjectId(app_id)},
            {"$set": {"status": status, "decision_reason": reason}}
        )
        return True
    except:
        return False

# ========================================
# 13. AUTH SCREEN
# ========================================
if st.session_state.user is None:
    st.markdown("""
    <div style='text-align: center; padding: 50px 0'>
        <h1>🏦 Microfinance Loan Default Prediction</h1>
        <p style='font-size: 18px; color: #666'>Smart Lending Decisions</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        st.divider()
        tab1, tab2 = st.tabs(["Login", "Register"])
        
        with tab1:
            st.subheader("🔐 Login")
            st.info("**Default Admin Account:**\n- Username: `admin`\n- Password: `admin123`")
            
            username = st.text_input("Username", key="login_user")
            password = st.text_input("Password", type="password", key="login_pass")
            
            if st.button("Login", use_container_width=True, type="primary"):
                if not username or not password:
                    st.error("❌ Enter username and password")
                else:
                    user = login(username, password)
                    if user == "PENDING":
                        st.warning("⏳ Account pending admin approval")
                    elif user:
                        st.session_state.user = user
                        st.success("✅ Login successful!")
                        st.rerun()
                    else:
                        st.error("❌ Invalid credentials")
        
        with tab2:
            st.subheader("📝 Register")
            username = st.text_input("Username", key="reg_user")
            password = st.text_input("Password", type="password", key="reg_pass")
            role = st.selectbox("Role", ["customer", "loan_officer", "risk_manager", "admin"], key="reg_role")
            
            if st.button("Register", use_container_width=True, type="primary"):
                if not username or not password:
                    st.error("❌ Enter username and password")
                elif len(password) < 6:
                    st.error("❌ Password must be 6+ characters")
                else:
                    result = register(username, password, role)
                    if result == "success":
                        st.success(f"✅ Account created!\nWait for admin approval.")
                        if role == "customer":
                            create_account(username, password)
                    elif result == "exists":
                        st.error("❌ Username already exists")
                    else:
                        st.error("❌ Registration error")
    st.stop()

# ========================================
# 14. MAIN APP (AFTER LOGIN)
# ========================================
user = st.session_state.user

# ========================================
# 15. ROLE-BASED ROUTING
# ========================================

if user['role'] == 'customer':
    # ===== CUSTOMER DASHBOARD =====
    
    with st.sidebar:
        st.markdown(f"<h3 style='color: white'>👤 {user['username'].upper()}</h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='color: white'>**Role:** Customer</p>", unsafe_allow_html=True)
        st.divider()
        
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.user = None
            st.session_state.models = None
            st.rerun()
    
    with st.sidebar:
        st.markdown("<h3 style='color: white'>📊 Navigation</h3>", unsafe_allow_html=True)
        page = st.radio(
            "Select Page",
            ["Dashboard", "Deposit", "Withdraw", "Transfer", "Card Recharge", "Apply for Loan", "Loan Status", "Withdraw Loan", "Transaction History"],
            label_visibility="collapsed"
        )
    
    # CUSTOMER PAGES
    
    if page == "Dashboard":
        st.title("💳 My Dashboard")
        st.divider()
        
        balance = get_account_balance(user['_id'])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("💰 Account Balance", f"${balance:,.2f}")
        with col2:
            st.metric("👤 Username", user['username'])
        with col3:
            loan_status = get_user_loan_status(user['_id'])
            status = loan_status['status'].upper() if loan_status else "NO APPLICATION"
            st.metric("📋 Loan Status", status)
        
        st.divider()
        st.markdown("### Quick Actions")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            if st.button("💵 Deposit", use_container_width=True):
                st.session_state.page = "Deposit"
                st.rerun()
        with col2:
            if st.button("💸 Withdraw", use_container_width=True):
                st.session_state.page = "Withdraw"
                st.rerun()
        with col3:
            if st.button("🔄 Transfer", use_container_width=True):
                st.session_state.page = "Transfer"
                st.rerun()
        with col4:
            if st.button("🎫 Recharge Card", use_container_width=True):
                st.session_state.page = "Card Recharge"
                st.rerun()
        with col5:
            if st.button("📊 Apply Loan", use_container_width=True):
                st.session_state.page = "Apply for Loan"
                st.rerun()
    
    elif page == "Deposit":
        st.title("💵 Deposit Money")
        st.divider()
        
        amount = st.number_input("Amount to Deposit ($)", min_value=1, step=100)
        
        if st.button("✅ Deposit", use_container_width=True, type="primary"):
            if deposit_money(user['_id'], amount):
                st.success(f"✅ Successfully deposited ${amount:,.2f}")
                st.rerun()
            else:
                st.error("❌ Deposit failed")
    
    elif page == "Withdraw":
        st.title("💸 Withdraw Money")
        st.divider()
        
        balance = get_account_balance(user['_id'])
        st.info(f"💰 Current Balance: ${balance:,.2f}")
        
        amount = st.number_input("Amount to Withdraw ($)", min_value=1, max_value=int(balance) if balance > 0 else 1, step=100)
        
        if st.button("✅ Withdraw", use_container_width=True, type="primary"):
            result = withdraw_money(user['_id'], amount)
            if result == "success":
                st.success(f"✅ Successfully withdrew ${amount:,.2f}")
                st.rerun()
            elif result == "insufficient":
                st.error("❌ Insufficient balance")
            elif result == "invalid":
                st.error("❌ Invalid amount")
            else:
                st.error("❌ Withdrawal failed")
    
    elif page == "Transfer":
        st.title("🔄 Transfer Money")
        st.divider()
        
        balance = get_account_balance(user['_id'])
        st.info(f"💰 Current Balance: ${balance:,.2f}")
        
        col1, col2 = st.columns(2)
        with col1:
            recipient = st.text_input("Recipient Username")
        with col2:
            amount = st.number_input("Amount ($)", min_value=1, max_value=int(balance) if balance > 0 else 1, step=100)
        
        if st.button("✅ Transfer", use_container_width=True, type="primary"):
            if not recipient:
                st.error("❌ Enter recipient username")
            else:
                result = transfer_money(user['_id'], recipient, amount)
                if result == "success":
                    st.success(f"✅ Successfully transferred ${amount:,.2f} to {recipient}")
                    st.rerun()
                elif result == "insufficient":
                    st.error("❌ Insufficient balance")
                elif result == "not_found":
                    st.error("❌ Recipient not found")
                elif result == "recipient_no_account":
                    st.error("❌ Recipient has no bank account")
                else:
                    st.error("❌ Transfer failed")
    
    elif page == "Card Recharge":
        st.title("🎫 Recharge Card")
        st.divider()
        
        balance = get_account_balance(user['_id'])
        st.info(f"💰 Current Balance: ${balance:,.2f}")
        
        col1, col2 = st.columns(2)
        with col1:
            card_number = st.text_input("Card Number (Last 4 digits visible)")
        with col2:
            amount = st.number_input("Recharge Amount ($)", min_value=1, max_value=int(balance) if balance > 0 else 1, step=100)
        
        if st.button("✅ Recharge", use_container_width=True, type="primary"):
            if len(card_number) < 4:
                st.error("❌ Invalid card number")
            else:
                result = recharge_card(user['_id'], amount, card_number)
                if result == "success":
                    st.success(f"✅ Card recharged successfully with ${amount:,.2f}")
                    st.rerun()
                elif result == "insufficient":
                    st.error("❌ Insufficient balance")
                else:
                    st.error("❌ Recharge failed")
    
    elif page == "Apply for Loan":
        st.title("📊 Apply for Loan")
        st.divider()
        
        balance = get_account_balance(user['_id'])
        txn_count = get_transaction_count(user['_id'], days=30)
        
        st.markdown("### 🤖 Smart Loan Offer Calculation")
        st.info(f"""
        **Your Profile:**
        - Current Balance: ${balance:,.2f}
        - Transactions (Last 30 days): {txn_count}
        """)
        
        # Calculate smart loan offer
        offered_amount = calculate_loan_amount(balance, txn_count, defaults=0, repayment_rate=80)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("💰 Offered Loan Amount", f"${offered_amount:,.2f}")
        with col2:
            st.metric("📈 Max Multiplier", "5x Account Balance")
        with col3:
            st.metric("📊 Activity Bonus", f"${min(txn_count * 1000, balance * 1.5):,.2f}")
        
        st.divider()
        st.markdown("### Accept Offer")
        
        col1, col2 = st.columns(2)
        with col1:
            accepted_amount = st.number_input(
                "Loan Amount You Want", 
                min_value=1000, 
                max_value=int(offered_amount),
                value=int(offered_amount),
                step=1000
            )
        with col2:
            duration = st.selectbox("Loan Duration (months)", [6, 12, 18, 24, 36])
        
        if st.button("✅ Apply for Loan", use_container_width=True, type="primary"):
            if apply_for_loan_user(user['_id'], user['username'], offered_amount):
                accept_loan_offer(user['_id'], accepted_amount, duration)
                st.success("✅ Loan application submitted!")
                st.info("⏳ Processing your application with AI model...")
                st.balloons()
                st.rerun()
            else:
                st.error("❌ Failed to submit application")
    
    elif page == "Loan Status":
        st.title("📋 Loan Application Status")
        st.divider()
        
        loan_status = get_user_loan_status(user['_id'])
        
        if loan_status:
            status_emoji = "🟢" if loan_status['status'] == 'approved' else ("🔴" if loan_status['status'] == 'rejected' else "🟡")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Status", f"{status_emoji} {loan_status['status'].upper()}")
            with col2:
                st.metric("Offered Amount", f"${loan_status['offered_amount']:,.2f}")
            with col3:
                st.metric("Duration", f"{loan_status['duration']} months")
            with col4:
                st.metric("Risk Score", f"{loan_status.get('risk_score', 'N/A')}")
            
            st.divider()
            st.markdown("### Decision Details")
            st.info(f"**Reason:** {loan_status['reason']}")
            
            # If rejected, show eligibility timeline
            if loan_status['status'] == 'rejected':
                st.divider()
                st.markdown("### ⏰ Path to Loan Eligibility")
                
                eligibility = predict_eligibility_timeline(loan_status.get('risk_score', 50), 80)
                if eligibility:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Eligible In", f"{eligibility['months_to_eligible']} months")
                        st.metric("Eligible Date", eligibility['eligible_date'].strftime("%Y-%m-%d"))
                    with col2:
                        st.metric("Suggested Loan Amount", f"${eligibility['suggested_loan_amount']:,.2f}")
                    
                    st.divider()
                    st.markdown("### 📋 How to Improve Your Application:")
                    for i, improvement in enumerate(eligibility['required_improvements'], 1):
                        st.write(f"{i}. ✅ {improvement}")
            
            # If approved, show repayment schedule
            if loan_status['status'] == 'approved' and loan_status['schedule']:
                st.divider()
                st.markdown("### 📅 Repayment Schedule")
                
                schedule = loan_status['schedule']
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Days Left to Pay", f"{loan_status['days_left']} days")
                with col2:
                    st.metric("Daily Payment", f"${schedule['daily']:.2f}")
                with col3:
                    st.metric("Weekly Payment", f"${schedule['weekly']:.2f}")
                with col4:
                    st.metric("Monthly Payment", f"${schedule['monthly']:.2f}")
                
                st.divider()
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Quarterly Payment", f"${schedule['quarterly']:.2f}")
                with col2:
                    st.metric("Total to Pay Back", f"${schedule['total_amount']:,.2f}")
        else:
            st.warning("❌ No loan applications found")
    
    elif page == "Withdraw Loan":
        st.title("💰 Withdraw Loan Money")
        st.divider()
        
        loan_status = get_user_loan_status(user['_id'])
        
        if not loan_status:
            st.error("❌ No loan application found")
        elif loan_status['status'] != 'approved':
            st.error(f"❌ Loan not approved. Current status: {loan_status['status'].upper()}")
        else:
            available = loan_status['accepted_amount'] - loan_status.get('borrowed_amount', 0)
            st.info(f"💰 Approved Amount: ${loan_status['accepted_amount']:,.2f}\n\n💵 Already Borrowed: ${loan_status.get('borrowed_amount', 0):,.2f}\n\n✅ Available to Withdraw: ${available:,.2f}")
            
            amount = st.number_input("Amount to Withdraw ($)", min_value=1, max_value=int(available) if available > 0 else 1, step=100)
            
            if st.button("✅ Withdraw Loan Money", use_container_width=True, type="primary"):
                result = withdraw_loan_money(user['_id'], amount)
                if result == "success":
                    st.success(f"✅ Successfully withdrew ${amount:,.2f} to your account")
                    st.rerun()
                elif result == "exceeds_limit":
                    st.error("❌ Amount exceeds approved limit")
                elif result == "not_approved":
                    st.error("❌ Loan not approved")
                else:
                    st.error("❌ Withdrawal failed")
    
    elif page == "Transaction History":
        st.title("📝 Transaction History")
        st.divider()
        
        transactions = get_user_transactions(user['_id'])
        
        if len(transactions) > 0:
            st.dataframe(
                transactions[['type', 'amount', 'timestamp', 'status']].sort_values('timestamp', ascending=False),
                use_container_width=True
            )
        else:
            st.info("No transactions yet")

else:
    # ===== STAFF DASHBOARD (Loan Officer, Risk Manager, Admin) =====
    
    with st.sidebar:
        st.markdown(f"<h3 style='color: white'>👤 {user['username'].upper()}</h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='color: white'>**Role:** {user['role'].replace('_', ' ').title()}</p>", unsafe_allow_html=True)
        st.divider()
        
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.user = None
            st.session_state.models = None
            st.rerun()

    with st.sidebar:
        st.markdown("<h3 style='color: white'>📊 Navigation</h3>", unsafe_allow_html=True)
        
        if user['role'] == 'admin':
            pages = ["Dashboard", "Risk Analysis", "Borrowers", "Models", "EDA", "Forecasting", "Loan Processing", "Decisions", "Customer Loan Requests", "Admin"]
        elif user['role'] == 'loan_officer':
            pages = ["Dashboard", "Risk Analysis", "Borrowers", "Models", "Loan Processing", "Decisions", "Customer Loan Requests"]
        else:  # risk_manager
            pages = ["Dashboard", "Risk Analysis", "Models", "EDA", "Forecasting", "Decisions", "Customer Loan Requests"]
        
        page = st.radio(
            "Select Page",
            pages,
            label_visibility="collapsed"
        )

    # ========================================
    # 16. STAFF PAGES
    # ========================================

    # DASHBOARD
    if page == "Dashboard":
        st.title("🏦 Dashboard")
        st.divider()
        
        loans = get_loans()
        borrowers = get_borrowers()
        
        if len(loans) > 0:
            approved = len(loans[loans["status"] == "approved"])
            rejected = len(loans[loans["status"] == "rejected"])
            pending = len(loans[loans["status"] == "pending_review"])
            total_exposure = loans[loans["status"] == "approved"]["amount"].sum()
        else:
            approved = rejected = pending = total_exposure = 0
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:
            st.metric("👥 Borrowers", len(borrowers))
        with col2:
            st.metric("📋 Loans", len(loans))
        with col3:
            st.metric("✅ Approved", approved)
        with col4:
            st.metric("❌ Rejected", rejected)
        with col5:
            st.metric("⏳ Pending", pending)
        with col6:
            st.metric("💰 Exposure", f"${total_exposure:,.0f}")
        
        st.divider()
        
        if len(loans) > 0:
            col1, col2 = st.columns(2)
            with col1:
                try:
                    status_counts = loans["status"].value_counts()
                    fig = px.pie(
                        values=status_counts.values,
                        names=status_counts.index,
                        title="Loan Status Distribution",
                        color_discrete_map={"approved": "#2ecc71", "rejected": "#e74c3c", "pending_review": "#f39c12"}
                    )
                    fig.update_layout(template="plotly_dark")
                    st.plotly_chart(fig, use_container_width=True)
                except:
                    pass
            
            with col2:
                try:
                    risk_data = loans['risk_score'].dropna()
                    if len(risk_data) > 0:
                        fig = px.histogram(
                            x=risk_data,
                            nbins=20,
                            title="Risk Score Distribution",
                            labels={"x": "Risk Score (%)", "count": "Count"},
                            color_discrete_sequence=["#3498db"]
                        )
                        fig.update_layout(template="plotly_dark")
                        st.plotly_chart(fig, use_container_width=True)
                except:
                    pass

    # RISK ANALYSIS
    elif page == "Risk Analysis":
        st.title("📊 Risk Analysis")
        st.divider()
        
        borrowers = get_borrowers()
        
        if len(borrowers) >= 5:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📉 Default Rate", f"{(borrowers['defaults'] > 0).mean():.2%}")
            with col2:
                st.metric("💰 Avg Income", f"${borrowers['income'].mean():,.0f}")
            with col3:
                st.metric("👤 Avg Age", f"{borrowers['age'].mean():.0f} years")
            
            st.divider()
            st.dataframe(borrowers[['name', 'income', 'age', 'repayment_history', 'defaults']], use_container_width=True)
        else:
            st.warning(f"Need 5+ borrowers. Current: {len(borrowers)}")

    # BORROWERS
    elif page == "Borrowers":
        st.title("👥 Borrower Management")
        st.divider()
        
        tab1, tab2 = st.tabs(["Add Single", "Bulk Upload"])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                name = st.text_input("Name")
                age = st.number_input("Age", 18, 100, 30)
                income = st.number_input("Annual Income", 5000, 1000000, 50000)
            with col2:
                repayment = st.number_input("Repayment History (%)", 0.0, 100.0, 80.0)
                prev_loans = st.number_input("Previous Loans", 0, 50, 0)
                defaults = st.number_input("Defaults", 0, 10, 0)
            
            if st.button("✅ Add Borrower", use_container_width=True, type="primary"):
                if not name or income <= 0:
                    st.error("❌ Invalid input")
                elif add_borrower(name, age, income, repayment, prev_loans, defaults):
                    st.success(f"✅ {name} added!")
                else:
                    st.error("❌ Error adding borrower")
        
        with tab2:
            uploaded = st.file_uploader("Upload CSV/Excel", type=['csv', 'xlsx'])
            if uploaded:
                try:
                    if uploaded.name.endswith('.csv'):
                        df = pd.read_csv(uploaded)
                    else:
                        df = pd.read_excel(uploaded)
                    
                    st.dataframe(df.head(), use_container_width=True)
                    
                    if st.button("📤 Upload", use_container_width=True, type="primary"):
                        count, errors = bulk_upload_borrowers(df)
                        st.success(f"✅ Added {count} borrowers")
                        if errors:
                            with st.expander("⚠️ Errors"):
                                for err in errors[:10]:
                                    st.text(err)
                except Exception as e:
                    st.error(f"Error: {e}")
        
        st.divider()
        borrowers = get_borrowers()
        st.write(f"**Total Borrowers:** {len(borrowers)}")
        if len(borrowers) > 0:
            st.dataframe(borrowers[['name', 'income', 'age', 'repayment_history']], use_container_width=True)

    # MODELS
    elif page == "Models":
        st.title("🤖 Model Training & Evaluation")
        st.divider()
        
        borrowers = get_borrowers()
        
        if len(borrowers) < 20:
            st.warning(f"Need 20+ borrowers. Current: {len(borrowers)}")
        else:
            if st.button("🚀 Train Models", use_container_width=True, type="primary"):
                with st.spinner("Training models..."):
                    models = train_models(borrowers)
                    if models:
                        st.session_state.models = models
                        st.success("✅ Training complete!")
                        st.info("🔄 Auto-processing customer loan applications...")
                        processed = auto_process_loan_applications()
                        st.success(f"✅ Auto-processed {processed} loan applications")
            
            if st.session_state.models:
                results = st.session_state.models
                
                st.markdown("### 📊 Performance Metrics Comparison")
                metrics_data = {
                    "Model": list(results.keys()),
                    "Accuracy": [f"{results[m]['Accuracy']:.4f}" for m in results.keys()],
                    "Precision": [f"{results[m]['Precision']:.4f}" for m in results.keys()],
                    "Recall": [f"{results[m]['Recall']:.4f}" for m in results.keys()],
                    "F1": [f"{results[m]['F1']:.4f}" for m in results.keys()],
                    "AUC": [f"{results[m]['AUC']:.4f}" for m in results.keys()],
                    "CV Mean": [f"{results[m]['CV_Mean']:.4f}" for m in results.keys()],
                    "CV Std": [f"{results[m]['CV_Std']:.4f}" for m in results.keys()]
                }
                
                st.dataframe(pd.DataFrame(metrics_data), use_container_width=True)
                
                best_model = max(results.items(), key=lambda x: x[1]["AUC"])[0]
                st.success(f"✅ **Best Model: {best_model}** (AUC: {results[best_model]['AUC']:.4f})")
                
                st.divider()
                st.markdown("### 📈 ROC Curves")
                
                fig_roc = go.Figure()
                for model_name in results.keys():
                    fpr = results[model_name]['fpr']
                    tpr = results[model_name]['tpr']
                    auc_score = results[model_name]['AUC']
                    fig_roc.add_trace(go.Scatter(
                        x=fpr, y=tpr,
                        mode='lines',
                        name=f'{model_name} (AUC: {auc_score:.4f})',
                        line=dict(width=2)
                    ))
                
                fig_roc.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1],
                    mode='lines',
                    name='Random Classifier',
                    line=dict(width=2, dash='dash', color='gray')
                ))
                
                fig_roc.update_layout(
                    title='ROC Curves Comparison',
                    xaxis_title='False Positive Rate',
                    yaxis_title='True Positive Rate',
                    template='plotly_dark',
                    hovermode='closest'
                )
                st.plotly_chart(fig_roc, use_container_width=True)
                
                st.divider()
                st.markdown("### 🔲 Confusion Matrices")
                
                col1, col2, col3 = st.columns(3)
                for idx, (model_name, model_info) in enumerate(results.items()):
                    with [col1, col2, col3][idx]:
                        cm = confusion_matrix(model_info['y_test'], model_info['y_pred'])
                        fig_cm = go.Figure(data=go.Heatmap(
                            z=cm,
                            x=['Predicted No Default', 'Predicted Default'],
                            y=['Actual No Default', 'Actual Default'],
                            colorscale='Blues',
                            text=cm,
                            texttemplate="%{text}",
                            textfont={"size": 14}
                        ))
                        fig_cm.update_layout(
                            title=f'{model_name}',
                            template='plotly_dark',
                            height=400
                        )
                        st.plotly_chart(fig_cm, use_container_width=True)

    # EDA
    elif page == "EDA":
        st.title("📊 Exploratory Data Analysis")
        st.divider()
        
        borrowers = get_borrowers()
        
        if len(borrowers) < 5:
            st.warning(f"Need 5+ borrowers. Current: {len(borrowers)}")
        else:
            st.markdown("### Statistical Summary")
            st.dataframe(borrowers[['income', 'age', 'repayment_history', 'previous_loans', 'transaction_freq', 'defaults']].describe(), use_container_width=True)
            
            st.divider()
            
            charts = generate_eda_charts(borrowers)
            
            if "scatter_income_defaults" in charts:
                st.markdown("### 📍 Scatter Plots")
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(charts["scatter_income_defaults"], use_container_width=True)
                with col2:
                    st.plotly_chart(charts["scatter_age_repayment"], use_container_width=True)
            
            st.divider()
            
            if "dist_income" in charts:
                st.markdown("### 📊 Distributions")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.plotly_chart(charts["dist_income"], use_container_width=True)
                with col2:
                    st.plotly_chart(charts["dist_age"], use_container_width=True)
                with col3:
                    st.plotly_chart(charts["dist_repayment"], use_container_width=True)
            
            st.divider()
            
            if "correlation" in charts:
                st.markdown("### 🔗 Feature Correlation")
                st.plotly_chart(charts["correlation"], use_container_width=True)
            
            st.divider()
            
            if "box_plot" in charts:
                st.markdown("### 📦 Box Plots")
                st.plotly_chart(charts["box_plot"], use_container_width=True)

    # FORECASTING
    elif page == "Forecasting":
        st.title("🔮 Loan Forecasting Analysis")
        st.divider()
        
        loans = get_loans()
        
        forecast_periods = st.slider("Forecast Periods (days)", 7, 90, 30)
        
        st.markdown("### 💰 Average Loan Amount Forecast")
        forecast_amount = forecast_loan_amounts(loans, periods=forecast_periods)
        
        if forecast_amount:
            forecast_df, historical, model = forecast_amount
            
            fig_amount = go.Figure()
            
            # Historical data
            fig_amount.add_trace(go.Scatter(
                x=historical.index,
                y=historical.values,
                mode='lines',
                name='Historical Average',
                line=dict(color='#3498db', width=2)
            ))
            
            # Forecast
            fig_amount.add_trace(go.Scatter(
                x=forecast_df.index,
                y=forecast_df['forecast'],
                mode='lines+markers',
                name='Forecast',
                line=dict(color='#2ecc71', width=2, dash='dash')
            ))
            
            # Confidence interval if available
            if 'lower' in forecast_df.columns and 'upper' in forecast_df.columns:
                fig_amount.add_trace(go.Scatter(
                    x=forecast_df.index,
                    y=forecast_df['upper'],
                    fill=None,
                    mode='lines',
                    line_color='rgba(0,0,0,0)',
                    showlegend=False
                ))
                
                fig_amount.add_trace(go.Scatter(
                    x=forecast_df.index,
                    y=forecast_df['lower'],
                    fill='tonexty',
                    mode='lines',
                    line_color='rgba(0,0,0,0)',
                    name='95% Confidence Interval',
                    fillcolor='rgba(46,204,113,0.2)'
                ))
            
            fig_amount.update_layout(
                title='Average Loan Amount Forecast',
                xaxis_title='Date',
                yaxis_title='Amount ($)',
                template='plotly_dark',
                hovermode='x unified'
            )
            st.plotly_chart(fig_amount, use_container_width=True)
            st.success(f"✅ Forecasted {len(forecast_df)} days of data")
        else:
            st.info("Add some loans to see forecasting")
        
        st.divider()
        
        st.markdown("### ⚠️ Default Risk Forecast")
        forecast_risk = forecast_default_risk(loans, periods=forecast_periods)
        
        if forecast_risk:
            forecast_df, historical, model = forecast_risk
            
            fig_risk = go.Figure()
            
            fig_risk.add_trace(go.Scatter(
                x=historical.index,
                y=historical.values,
                mode='lines',
                name='Historical Risk',
                line=dict(color='#e74c3c', width=2)
            ))
            
            fig_risk.add_trace(go.Scatter(
                x=forecast_df.index,
                y=forecast_df['forecast'],
                mode='lines+markers',
                name='Risk Forecast',
                line=dict(color='#f39c12', width=2, dash='dash')
            ))
            
            if 'lower' in forecast_df.columns and 'upper' in forecast_df.columns:
                fig_risk.add_trace(go.Scatter(
                    x=forecast_df.index,
                    y=forecast_df['upper'],
                    fill=None,
                    mode='lines',
                    line_color='rgba(0,0,0,0)',
                    showlegend=False
                ))
                
                fig_risk.add_trace(go.Scatter(
                    x=forecast_df.index,
                    y=forecast_df['lower'],
                    fill='tonexty',
                    mode='lines',
                    line_color='rgba(0,0,0,0)',
                    name='95% Confidence Interval',
                    fillcolor='rgba(243,156,18,0.2)'
                ))
            
            fig_risk.update_layout(
                title='Default Risk Score Forecast',
                xaxis_title='Date',
                yaxis_title='Risk Score (%)',
                template='plotly_dark',
                hovermode='x unified'
            )
            st.plotly_chart(fig_risk, use_container_width=True)
            st.success(f"✅ Forecasted {len(forecast_df)} days of risk data")
        else:
            st.info("Add some loans to see risk forecasting")
        
        st.divider()
        
        st.markdown("### ✅ Approval Rate Forecast")
        forecast_approval = forecast_approval_rate(loans, periods=forecast_periods)
        
        if forecast_approval:
            forecast_df, historical, model = forecast_approval
            
            fig_approval = go.Figure()
            
            fig_approval.add_trace(go.Scatter(
                x=historical.index,
                y=historical.values,
                mode='lines',
                name='Historical Approval Rate',
                line=dict(color='#2ecc71', width=2)
            ))
            
            fig_approval.add_trace(go.Scatter(
                x=forecast_df.index,
                y=forecast_df['forecast'],
                mode='lines+markers',
                name='Approval Rate Forecast',
                line=dict(color='#9b59b6', width=2, dash='dash')
            ))
            
            if 'lower' in forecast_df.columns and 'upper' in forecast_df.columns:
                fig_approval.add_trace(go.Scatter(
                    x=forecast_df.index,
                    y=forecast_df['upper'],
                    fill=None,
                    mode='lines',
                    line_color='rgba(0,0,0,0)',
                    showlegend=False
                ))
                
                fig_approval.add_trace(go.Scatter(
                    x=forecast_df.index,
                    y=forecast_df['lower'],
                    fill='tonexty',
                    mode='lines',
                    line_color='rgba(0,0,0,0)',
                    name='95% Confidence Interval',
                    fillcolor='rgba(155,89,182,0.2)'
                ))
            
            fig_approval.update_layout(
                title='Loan Approval Rate Forecast',
                xaxis_title='Date',
                yaxis_title='Approval Rate (%)',
                template='plotly_dark',
                hovermode='x unified',
                yaxis=dict(range=[0, 100])
            )
            st.plotly_chart(fig_approval, use_container_width=True)
            st.success(f"✅ Forecasted {len(forecast_df)} days of approval data")
        else:
            st.info("Add some loans to see approval rate forecasting")

    # LOAN PROCESSING
    elif page == "Loan Processing":
        st.title("💳 Loan Processing")
        st.divider()
        
        borrowers = get_borrowers()
        
        if len(borrowers) == 0:
            st.warning("No borrowers available")
        else:
            selected = st.selectbox(
                "Select Borrower",
                borrowers['name'].tolist(),
                format_func=lambda x: x
            )
            
            borrower = borrowers[borrowers['name'] == selected].iloc[0].to_dict()
            
            st.markdown("### Borrower Profile")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Age", borrower['age'])
            with col2:
                st.metric("Income", f"${borrower['income']:,.0f}")
            with col3:
                st.metric("Repayment", f"{borrower['repayment_history']:.1f}%")
            with col4:
                st.metric("Defaults", borrower['defaults'])
            
            st.divider()
            
            if not st.session_state.models:
                st.warning("⚠️ Train models first!")
            else:
                models = st.session_state.models
                model_name = st.selectbox("Select Model", list(models.keys()))
                model_info = models[model_name]
                
                if st.button("🔮 Process Application", use_container_width=True, type="primary"):
                    data = [
                        safe_num(borrower['income'], is_float=True),
                        safe_num(borrower['age']),
                        safe_num(borrower['repayment_history'], is_float=True),
                        safe_num(borrower['previous_loans']),
                        safe_num(borrower['transaction_freq'], is_float=True)
                    ]
                    
                    risk = predict_risk(
                        model_info['model'],
                        model_info['scaler'],
                        data,
                        model_name
                    )
                    
                    decision, reason = get_loan_decision(
                        risk,
                        safe_num(borrower['repayment_history'], is_float=True)
                    )
                    
                    if decision == "APPROVE":
                        amount = calculate_loan_amount(borrower['income'], borrower['transaction_freq'])
                        duration = 12 if amount < borrower['income'] else 18
                    else:
                        amount = 0
                        duration = 0
                    
                    if create_loan(
                        borrower['_id'],
                        amount,
                        duration,
                        risk,
                        model_name,
                        decision,
                        reason
                    ):
                        color = "🟢" if decision == "APPROVE" else ("🔴" if decision == "REJECT" else "🟡")
                        st.success(f"{color} **Decision: {decision}**")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Risk Score", f"{risk:.2f}%")
                        with col2:
                            st.metric("Loan Amount", f"${amount:,.0f}")
                        with col3:
                            st.metric("Duration", f"{duration} months")
                        with col4:
                            st.metric("Reason", reason.split(':')[1].strip() if ':' in reason else reason)

    # DECISIONS
    elif page == "Decisions":
        st.title("✅ Loan Decisions History")
        st.divider()
        
        decisions = get_loan_decisions()
        
        if len(decisions) > 0:
            approved = len(decisions[decisions['status'] == 'approved'])
            rejected = len(decisions[decisions['status'] == 'rejected'])
            pending = len(decisions[decisions['status'] == 'pending_review'])
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total", len(decisions))
            with col2:
                st.metric("✅ Approved", approved)
            with col3:
                st.metric("❌ Rejected", rejected)
            with col4:
                st.metric("⏳ Pending", pending)
            
            st.divider()
            st.dataframe(
                decisions[['borrower_name', 'amount', 'risk', 'status', 'reason']],
                use_container_width=True
            )
        else:
            st.info("No loan decisions yet")

    # CUSTOMER LOAN REQUESTS
    elif page == "Customer Loan Requests":
        st.title("📋 Customer Loan Applications")
        st.divider()
        
        apps = get_all_loan_applications()
        
        if len(apps) > 0:
            approved = len(apps[apps['status'] == 'approved'])
            rejected = len(apps[apps['status'] == 'rejected'])
            pending = len(apps[apps['status'] == 'pending'])
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Applications", len(apps))
            with col2:
                st.metric("✅ Approved", approved)
            with col3:
                st.metric("❌ Rejected", rejected)
            with col4:
                st.metric("⏳ Pending", pending)
            
            st.divider()
            
            st.markdown("### All Applications")
            apps_display = apps[['username', 'offered_amount', 'accepted_amount', 'duration', 'status', 'decision_reason', 'created_at']].copy()
            st.dataframe(apps_display, use_container_width=True)
            
            st.divider()
            
            st.markdown("### Manage Applications")
            selected_app = st.selectbox(
                "Select Application",
                apps['username'].tolist(),
                format_func=lambda x: f"{x} - {apps[apps['username']==x].iloc[0]['status'].upper()}"
            )
            
            if selected_app:
                app_data = apps[apps['username'] == selected_app].iloc[0]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Offered Amount", f"${app_data['offered_amount']:,.2f}")
                with col2:
                    st.metric("Accepted Amount", f"${app_data.get('accepted_amount', 0):,.2f}")
                with col3:
                    st.metric("Current Status", app_data['status'].upper())
                
                st.divider()
                
                new_status = st.selectbox("New Status", ["approved", "rejected", "pending"])
                reason = st.text_area("Decision Reason", value=app_data.get('decision_reason', ''))
                
                if st.button("✅ Update Status", use_container_width=True, type="primary"):
                    if update_loan_application_status(app_data['_id'], new_status, reason):
                        st.success(f"✅ Updated {selected_app}'s application to {new_status.upper()}")
                        st.rerun()
                    else:
                        st.error("❌ Failed to update")
        else:
            st.info("No customer loan applications yet")

    # ADMIN
    elif page == "Admin":
        if user['role'] != 'admin':
            st.error("❌ Admin access required")
        else:
            st.title("⚙️ Admin Panel")
            st.divider()
            
            tab1, tab2, tab3, tab4 = st.tabs(["User Management", "User Approval", "System Info", "System Settings"])
            
            with tab1:
                st.markdown("### All Users in System")
                users = get_all_users()
                
                if len(users) > 0:
                    st.dataframe(users[['username', 'password', 'role', 'status', 'created_at']], use_container_width=True)
                    
                    st.divider()
                    st.markdown("### Delete Users")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        selected_user = st.selectbox("Select User to Delete", users['username'].tolist())
                        
                        if st.button("🗑️ Delete This User", use_container_width=True, type="secondary"):
                            user_id = users[users['username'] == selected_user]['_id'].values[0]
                            if delete_user(user_id):
                                st.success(f"✅ User {selected_user} deleted")
                                st.rerun()
                            else:
                                st.error("❌ Failed to delete user")
                    
                    with col2:
                        if st.button("🗑️ DELETE ALL USERS & TRANSACTIONS", use_container_width=True, type="secondary"):
                            if st.checkbox("⚠️ I confirm to delete all users and transactions"):
                                if delete_all_users_and_transactions():
                                    st.success("✅ All users and transactions deleted")
                                    st.rerun()
                else:
                    st.info("No users in system")
            
            with tab2:
                st.markdown("### Pending Approvals")
                users = get_pending_users()
                
                if len(users) > 0:
                    st.dataframe(users[['username', 'role', 'created_at']], use_container_width=True)
                    
                    for idx, user_row in users.iterrows():
                        if st.button(f"✅ Approve {user_row['username']}", key=f"approve_{idx}"):
                            if approve_user(user_row['_id']):
                                if user_row['role'] == 'customer':
                                    create_account(user_row['username'], "")
                                st.success("User approved!")
                                st.rerun()
                else:
                    st.info("No pending users")
            
            with tab3:
                st.markdown("### System Statistics")
                borrowers = get_borrowers()
                loans = get_loans()
                users = list(db['users'].find())
                accounts = list(db['accounts'].find())
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("👥 Borrowers", len(borrowers))
                with col2:
                    st.metric("📋 Loans", len(loans))
                with col3:
                    st.metric("👤 Users", len(users))
                with col4:
                    st.metric("💳 Accounts", len(accounts))
                
                st.divider()
                
                if len(accounts) > 0:
                    total_balance = sum([acc.get('balance', 0) for acc in accounts])
                    st.metric("💰 Total System Balance", f"${total_balance:,.2f}")
            
            with tab4:
                st.markdown("### System Configuration")
                st.info("""
                **Risk Assessment Thresholds:**
                - Low Risk: 0-40%
                - Medium Risk: 40-60%
                - High Risk: 60%+
                
                **Loan Calculation:**
                - Base: 3-5x Account Balance
                - Activity Bonus: Transaction Frequency × $1000
                - Minimum: $5,000
                - Maximum: 5x Account Balance
                
                **System Requirements:**
                - Minimum borrowers for training: 20
                - Forecasting: Works with 1+ transaction
                - Cross-validation folds: 5
                """)

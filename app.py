"""
🏦 MICROFINANCE LOAN DEFAULT PREDICTION SYSTEM
Brand New Streamlit App - Fully Optimized
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

warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# ========================================
# 1. PAGE CONFIG
# ========================================

# ===== UI =====

st.markdown("""<style>
.stApp { background: linear-gradient(to right, #007BFF, #FFC107, #FF0000); }
section[data-testid="stSidebar"] { background: black!important; }
.card { background: white; padding: 12px; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.08); margin-bottom: 8px; }
.alert-warning { background-color: #fff3cd; border-left: 4px solid #ffc107; padding: 12px; margin: 10px 0; }
.alert-danger { background-color: #f8d7da; border-left: 4px solid #dc3545; padding: 12px; margin: 10px 0; }
.alert-info { background-color: #d1ecf1; border-left: 4px solid #17a2b8; padding: 12px; margin: 10px 0; }
.alert-success { background-color: #d4edda; border-left: 4px solid #28a745; padding: 12px; margin: 10px 0; }
</style>""", unsafe_allow_html=True)


# ========================================
# 2. MONGODB CONNECTION - SINGLE & FAST
# ========================================
MONGODB_URI = "mongodb+srv://euawari_db_user:6SnKvQvXXzrGeypA@cluster0.fkkzcvz.mongodb.net/microfinance_db?retryWrites=true&w=majority"

@st.cache_resource
def init_mongodb():
    """Initialize MongoDB connection - ONCE"""
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
def calculate_loan_amount(borrower, risk_score):
    """Calculate loan amount based on profile"""
    income = safe_num(borrower.get("income"), is_float=True)
    age = safe_num(borrower.get("age"))
    repayment = safe_num(borrower.get("repayment_history"), is_float=True)
    prev_loans = safe_num(borrower.get("previous_loans"))
    defaults = safe_num(borrower.get("defaults"))
    txn_freq = safe_num(borrower.get("transaction_freq"), is_float=True)

    base = income * 0.5
    repayment_score = repayment / 100
    txn_score = min(txn_freq / 50, 1)
    experience_score = min(prev_loans / 10, 1)
    behavior = (repayment_score * 0.5) + (txn_score * 0.3) + (experience_score * 0.2)
    
    default_penalty = max(0.2, 1 - (defaults * 0.25))
    risk_penalty = max(0.1, 1 - risk_score / 100)
    age_factor = 0.8 if age < 25 else (1.0 if age < 60 else 0.7)

    amount = base * behavior * default_penalty * risk_penalty * age_factor
    return max(10000, min(amount, income * 1.5))

def get_loan_decision(risk_score, repayment, defaults):
    """Get loan decision"""
    if defaults > 0:
        return "REJECT", f"Previous defaults: {defaults}"
    if risk_score >= 60:
        return "REJECT", f"High risk: {risk_score:.2f}%"
    if risk_score >= 30:
        return "REVIEW", f"Medium risk: {risk_score:.2f}%"
    return "APPROVE", f"Low risk: {risk_score:.2f}%"

def create_loan(bid, amount, duration, risk, model, decision, reason):
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
            "actual_default": 0,
            "created_at": datetime.now()
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
# 8. MODEL TRAINING
# ========================================
def train_models(borrowers_df):
    """Train ML models"""
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
            else:
                model.fit(X_train_smote, y_train_smote)
                pred = model.predict(X_test)
                proba = model.predict_proba(X_test)[:, 1]
            
            results[model_name] = {
                "Accuracy": accuracy_score(y_test, pred),
                "Precision": precision_score(y_test, pred, zero_division=0),
                "Recall": recall_score(y_test, pred, zero_division=0),
                "F1": f1_score(y_test, pred, zero_division=0),
                "AUC": roc_auc_score(y_test, proba),
                "model": model,
                "scaler": scaler if model_name == "Logistic Regression" else None,
                "y_test": y_test,
                "y_pred": pred,
                "y_prob": proba
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

# ========================================
# 9. USER MANAGEMENT (ADMIN)
# ========================================
def get_pending_users():
    """Get pending user approvals"""
    try:
        users = list(db['users'].find({"status": "pending"}))
        return pd.DataFrame(users) if users else pd.DataFrame()
    except:
        return pd.DataFrame()

def approve_user(user_id):
    """Approve pending user"""
    try:
        db['users'].update_one({"_id": ObjectId(user_id)}, {"$set": {"status": "approved"}})
        return True
    except:
        return False

# ========================================
# 10. AUTH SCREEN
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
            role = st.selectbox("Role", ["loan_officer", "risk_manager", "admin"], key="reg_role")
            
            if st.button("Register", use_container_width=True, type="primary"):
                if not username or not password:
                    st.error("❌ Enter username and password")
                elif len(password) < 6:
                    st.error("❌ Password must be 6+ characters")
                else:
                    result = register(username, password, role)
                    if result == "success":
                        st.success(f"✅ Account created!\nWait for admin approval.")
                    elif result == "exists":
                        st.error("❌ Username already exists")
                    else:
                        st.error("❌ Registration error")
    st.stop()

# ========================================
# 11. MAIN APP (AFTER LOGIN)
# ========================================
user = st.session_state.user

# Sidebar
with st.sidebar:
    st.markdown(f"### 👤 {user['username'].upper()}")
    st.markdown(f"**Role:** {user['role']}")
    st.divider()
    
    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.user = None
        st.session_state.models = None
        st.rerun()

with st.sidebar:
    st.markdown("### 📊 Navigation")
    page = st.radio(
        "Select Page",
        ["Dashboard", "Risk Analysis", "Borrowers", "Models", "Loan Processing", "Decisions", "Admin"]
    )

# ========================================
# 12. PAGES
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
        total_exposure = loans[loans["status"] == "approved"]["amount"].sum()
    else:
        approved = rejected = total_exposure = 0
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("👥 Borrowers", len(borrowers))
    with col2:
        st.metric("📋 Loans", len(loans))
    with col3:
        st.metric("✅ Approved", approved)
    with col4:
        st.metric("❌ Rejected", rejected)
    with col5:
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
                    title="Loan Status",
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
            st.metric("📉 Default Rate", f"{borrowers['defaults'].mean():.2%}")
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
            repayment = st.number_input("Repayment History (%)", 0, 100, 80)
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
    st.title("🤖 Model Training")
    st.divider()
    
    borrowers = get_borrowers()
    
    if len(borrowers) < 20:
        st.warning(f"Need 20+ borrowers. Current: {len(borrowers)}")
    else:
        if st.button("🚀 Train Models", use_container_width=True, type="primary"):
            with st.spinner("Training..."):
                models = train_models(borrowers)
                if models:
                    st.session_state.models = models
                    st.success("✅ Training complete!")
        
        if st.session_state.models:
            results = st.session_state.models
            
            st.markdown("### Performance Metrics")
            metrics_data = {
                "Model": list(results.keys()),
                "Accuracy": [f"{results[m]['Accuracy']:.4f}" for m in results.keys()],
                "Precision": [f"{results[m]['Precision']:.4f}" for m in results.keys()],
                "Recall": [f"{results[m]['Recall']:.4f}" for m in results.keys()],
                "F1": [f"{results[m]['F1']:.4f}" for m in results.keys()],
                "AUC": [f"{results[m]['AUC']:.4f}" for m in results.keys()]
            }
            
            st.dataframe(pd.DataFrame(metrics_data), use_container_width=True)
            
            best_model = max(results.items(), key=lambda x: x[1]["AUC"])[0]
            st.info(f"✅ Best Model: **{best_model}** (AUC: {results[best_model]['AUC']:.4f})")

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
                    safe_num(borrower['repayment_history'], is_float=True),
                    safe_num(borrower['defaults'])
                )
                
                if decision == "APPROVE":
                    amount = calculate_loan_amount(borrower, risk)
                    duration = 12 if amount < borrower['income'] else 18
                else:
                    amount = 0
                    duration = 0
                
                # Save loan
                if create_loan(
                    borrower['_id'],
                    amount,
                    duration,
                    risk,
                    model_name,
                    decision,
                    reason
                ):
                    st.success(f"✅ Decision: **{decision}**")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Risk Score", f"{risk:.2f}%")
                    with col2:
                        st.metric("Loan Amount", f"${amount:,.0f}")
                    with col3:
                        st.metric("Duration", f"{duration} months")
                    with col4:
                        st.metric("Reason", reason.split(':')[1] if ':' in reason else reason)

# DECISIONS
elif page == "Decisions":
    st.title("✅ Loan Decisions")
    st.divider()
    
    decisions = get_loan_decisions()
    
    if len(decisions) > 0:
        approved = len(decisions[decisions['status'] == 'approved'])
        rejected = len(decisions[decisions['status'] == 'rejected'])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total", len(decisions))
        with col2:
            st.metric("✅ Approved", approved)
        with col3:
            st.metric("❌ Rejected", rejected)
        
        st.divider()
        st.dataframe(
            decisions[['borrower_name', 'amount', 'risk', 'status', 'reason']],
            use_container_width=True
        )
    else:
        st.info("No loan decisions yet")

# ADMIN
elif page == "Admin":
    if user['role'] != 'admin':
        st.error("❌ Admin access required")
    else:
        st.title("⚙️ Admin Panel")
        st.divider()
        
        tab1, tab2 = st.tabs(["User Approval", "System Info"])
        
        with tab1:
            st.markdown("### Pending Approvals")
            users = get_pending_users()
            
            if len(users) > 0:
                st.dataframe(users[['username', 'role', 'created_at']], use_container_width=True)
                
                for idx, user_row in users.iterrows():
                    if st.button(f"✅ Approve {user_row['username']}", key=f"approve_{idx}"):
                        if approve_user(user_row['_id']):
                            st.success("User approved!")
                            st.rerun()
            else:
                st.info("No pending users")
        
        with tab2:
            st.markdown("### System Statistics")
            borrowers = get_borrowers()
            loans = get_loans()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Borrowers", len(borrowers))
            with col2:
                st.metric("Loans", len(loans))
            with col3:
                st.metric("Users", len(list(db['users'].find())))

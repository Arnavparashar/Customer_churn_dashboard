import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Modern Professional Color Scheme
st.markdown("""
    <style>
    :root {
        --primary: #2563eb;       /* Vibrant blue */
        --secondary: #1e40af;    /* Darker blue */
        --accent: #3b82f6;       /* Light blue */
        --success: #10b981;      /* Emerald green */
        --danger: #ef4444;       /* Coral red */
        --warning: #f59e0b;      /* Amber */
        --light: #f8fafc;        /* Snow white */
        --dark: #1e293b;         /* Dark navy */
        --gray: #64748b;         /* Cool gray */
        --border-radius: 10px;
    }
    
    .main {
        background-color: #f1f5f9;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        line-height: 1.6;
    }
    
    .header-container {
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        padding: 2rem 1rem;
        border-radius: var(--border-radius);
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
    }
    
    .header {
        color: white;
        text-align: center;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .subheader {
        color: rgba(255,255,255,0.9);
        text-align: center;
        font-size: 1.1rem;
        max-width: 700px;
        margin: 0 auto;
    }
    
    .section {
        background-color: white;
        border-radius: var(--border-radius);
        padding: 1.75rem;
        margin-bottom: 1.75rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        border: 1px solid #e2e8f0;
    }
    
    .section-header {
        color: var(--dark);
        font-weight: 600;
        font-size: 1.25rem;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    .stSelectbox, .stSlider, .stNumberInput {
        border-radius: var(--border-radius) !important;
        border: 1px solid #cbd5e1 !important;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        color: white;
        border: none;
        border-radius: var(--border-radius);
        padding: 0.75rem 2rem;
        font-weight: 500;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 10px 15px -3px rgba(37, 99, 235, 0.3);
    }
    
    .prediction-card {
        padding: 2rem;
        border-radius: var(--border-radius);
        background: white;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        text-align: center;
        border-top: 4px solid;
    }
    
    .risk-high {
        border-color: var(--danger);
        background-color: #fef2f2;
    }
    
    .risk-low {
        border-color: var(--success);
        background-color: #f0fdf4;
    }
    
    .probability-meter {
        height: 10px;
        background: #e2e8f0;
        border-radius: 5px;
        margin: 1.5rem 0;
        overflow: hidden;
    }
    
    .meter-fill {
        height: 100%;
        background: linear-gradient(90deg, var(--danger), var(--warning));
        border-radius: 5px;
    }
    
    .metric-card {
        padding: 1.5rem 1rem;
        border-radius: var(--border-radius);
        background: white;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        text-align: center;
        border-top: 3px solid var(--accent);
    }
    
    .metric-value {
        font-size: 1.75rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .footer {
        text-align: center;
        color: var(--gray);
        font-size: 0.85rem;
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid #e2e8f0;
    }
    
    .icon {
        background: var(--primary);
        color: white;
        width: 36px;
        height: 36px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1rem;
    }
    
    .recommendations {
        background-color: #f8fafc;
        border-left: 4px solid var(--accent);
        border-radius: var(--border-radius);
        padding: 1.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Load the model and encoders
@st.cache_resource
def load_model_and_encoders():
    with open("customer_churn_model.pkl", "rb") as f:
        model_data = pickle.load(f)
        model = model_data['model'] if isinstance(model_data, dict) else model_data

    with open("encoders.pkl", "rb") as f:
        encoders = pickle.load(f)

    return model, encoders

model, encoders = load_model_and_encoders()

# Modern Header with Gradient Background
st.markdown("""
    <div class="header-container">
        <h1 class="header">Merchant Churn Prediction Analytics</h1>
        <p class="subheader">
            Advanced predictive analytics to identify at-risk merchants and optimize retention strategies
        </p>
    </div>
    """, unsafe_allow_html=True)

# Customer Details Section
with st.container():
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header"><div class="icon">👤</div> Merchant Profile</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        SeniorCitizen = st.selectbox("Senior Citizen", ["Yes", "No"])
        Partner = st.selectbox("Business Partner", ["Yes", "No"])
        Dependents = st.selectbox("Dependents", ["Yes", "No"])
    
    with col2:
        tenure = st.slider("Tenure (months)", 0, 72, 12,
                          help="Duration of merchant's relationship with your company")
        Contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    st.markdown('</div>', unsafe_allow_html=True)

# Billing Information Section
with st.container():
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header"><div class="icon">💳</div> Financial Metrics</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        MonthlyCharges = st.number_input("Monthly Charges ($)", min_value=0.0, value=70.0, step=5.0,
                                        help="Average monthly revenue from this merchant")
        TotalCharges = st.number_input("Total Charges ($)", min_value=0.0, value=1000.0, step=50.0,
                                     help="Lifetime value of this merchant")
    
    with col2:
        PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
        PaymentMethod = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Average Monthly Spend</div>', unsafe_allow_html=True)
        if tenure > 0:
            avg_spend = TotalCharges / tenure
            st.markdown(f'<div class="metric-value">${avg_spend:.2f}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="metric-value">$0.00</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Services Section
with st.container():
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header"><div class="icon">🛠️</div> Service Utilization</h2>', unsafe_allow_html=True)
    
    services_col1, services_col2, services_col3 = st.columns(3)
    
    with services_col1:
        PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
        MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    
    with services_col2:
        OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    
    with services_col3:
        TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    st.markdown('</div>', unsafe_allow_html=True)

# Prediction Button
if st.button("Analyze Churn Risk", use_container_width=True, key="predict_btn"):
    with st.spinner("Analyzing merchant data..."):
        try:
            # Collect input into DataFrame
            input_data = {
                'gender': gender,
                'SeniorCitizen': 1 if SeniorCitizen == "Yes" else 0,
                'Partner': Partner,
                'Dependents': Dependents,
                'tenure': tenure,
                'PhoneService': PhoneService,
                'MultipleLines': MultipleLines,
                'InternetService': InternetService,
                'OnlineSecurity': OnlineSecurity,
                'OnlineBackup': OnlineBackup,
                'DeviceProtection': DeviceProtection,
                'TechSupport': TechSupport,
                'StreamingTV': StreamingTV,
                'StreamingMovies': StreamingMovies,
                'Contract': Contract,
                'PaperlessBilling': PaperlessBilling,
                'PaymentMethod': PaymentMethod,
                'MonthlyCharges': MonthlyCharges,
                'TotalCharges': TotalCharges
            }

            input_df = pd.DataFrame([input_data])

            # Encode categorical columns
            for column, encoder in encoders.items():
                if column in input_df.columns:
                    input_df[column] = encoder.transform(input_df[[column]])

            # Make prediction
            prediction = model.predict(input_df)[0]
            prob = model.predict_proba(input_df)[0]
            
            # Display results
            st.markdown('<div style="margin-top: 2rem;">', unsafe_allow_html=True)
            
            # Risk card
            risk_class = "risk-high" if prediction == 1 else "risk-low"
            risk_text = "High Risk" if prediction == 1 else "Low Risk"
            risk_icon = "⚠️" if prediction == 1 else "✅"
            risk_color = "var(--danger)" if prediction == 1 else "var(--success)"
            
            st.markdown(f"""
                <div class="prediction-card {risk_class}">
                    <h2 style="margin-bottom: 0.75rem; color: {risk_color};">{risk_icon} {risk_text}</h2>
                    <p style="color: var(--gray); margin-bottom: 0.5rem;">Based on comprehensive merchant profile analysis</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Probability meter
            st.markdown(f"""
                <div class="section" style="margin-top: 1.5rem;">
                    <h3 style="color: var(--dark); margin-bottom: 1rem;">Churn Probability</h3>
                    <div class="probability-meter">
                        <div class="meter-fill" style="width: {prob[1]*100}%"></div>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-top: 0.5rem;">
                        <span style="font-size: 0.85rem; color: var(--gray);">0%</span>
                        <span style="font-size: 0.85rem; color: var(--gray);">50%</span>
                        <span style="font-size: 0.85rem; color: var(--gray);">100%</span>
                    </div>
                    <div style="text-align: center; margin-top: 0.5rem;">
                        <span style="font-size: 1rem; font-weight: 500; color: var(--dark);">{prob[1]*100:.1f}% probability</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Metrics cards
            cols = st.columns(2)
            with cols[0]:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Churn Probability</div>
                        <div class="metric-value" style="color: var(--danger);">{prob[1]*100:.1f}%</div>
                        <div style="font-size: 0.85rem; color: var(--gray);">Likelihood to churn</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            with cols[1]:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Retention Probability</div>
                        <div class="metric-value" style="color: var(--success);">{prob[0]*100:.1f}%</div>
                        <div style="font-size: 0.85rem; color: var(--gray);">Likelihood to stay</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Recommendations based on risk level
            if prediction == 1:
                st.markdown("""
                    <div class="section" style="margin-top: 1.5rem;">
                        <h3 style="color: var(--danger); margin-bottom: 1rem;">📌 Recommended Retention Actions</h3>
                        <div class="recommendations">
                            <ul style="color: var(--dark); padding-left: 1.25rem;">
                                <li style="margin-bottom: 0.75rem;">Offer loyalty discount or contract extension</li>
                                <li style="margin-bottom: 0.75rem;">Assign dedicated account manager for outreach</li>
                                <li style="margin-bottom: 0.75rem;">Analyze service usage for optimization</li>
                                <li>Schedule proactive check-in within 48 hours</li>
                            </ul>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div class="section" style="margin-top: 1.5rem; border-left: 4px solid var(--success);">
                        <h3 style="color: var(--success); margin-bottom: 1rem;">✅ Merchant Health Status</h3>
                        <p style="color: var(--dark);">This merchant shows strong engagement indicators. Consider:</p>
                        <div class="recommendations">
                            <ul style="color: var(--dark); padding-left: 1.25rem;">
                                <li style="margin-bottom: 0.75rem;">Upsell opportunities based on usage patterns</li>
                                <li style="margin-bottom: 0.75rem;">Referral program enrollment</li>
                                <li>Premium service recommendations</li>
                            </ul>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")

# Professional Footer
st.markdown("""
    <div class="footer">
        <div style="margin-bottom: 0.75rem;">
            <span style="color: var(--primary); font-weight: 600;">Merchant Retention Analytics</span> • <span style="color: var(--gray);">v2.2.0</span>
        </div>
        <div style="color: var(--gray); font-size: 0.8rem;">
            © 2024 Strategic Analytics Group | Confidential Business Intelligence
        </div>
    </div>
    """, unsafe_allow_html=True)
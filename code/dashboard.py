import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np

import os

# Get the directory where this file is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load datasets using absolute paths
df_raw = pd.read_csv(os.path.join(BASE_DIR, '../data/WA_Fn-UseC_-Telco-Customer-Churn.csv'))
df = pd.read_csv(os.path.join(BASE_DIR, '../data/telco_cleaned.csv'))

# Train the model
X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Dashboard title
st.title('Customer Churn Prediction Dashboard')
st.write('Customer Retention Tool')
st.markdown('---')

# ----------------------------------------
# Section 1: Churn Overview
# ----------------------------------------
st.header('Churn Overview')

col1, col2, col3 = st.columns(3)
with col1:
    total = len(df_raw)
    st.metric('Total Customers', f'{total:,}')
with col2:
    churned = (df_raw['Churn'] == 'Yes').sum()
    st.metric('Churned Customers', f'{churned:,}')
with col3:
    churn_rate = (churned / total) * 100
    st.metric('Churn Rate', f'{churn_rate:.1f}%')

st.markdown('---')

# ----------------------------------------
# Section 2: Visualizations
# ----------------------------------------
st.header('Data Visualizations')

tab1, tab2, tab3 = st.tabs([
    'Churn Distribution',
    'Churn by Contract',
    'Monthly Charges'
])

with tab1:
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    churn_counts = df_raw['Churn'].value_counts()
    ax1.pie(churn_counts,
            labels=['No Churn', 'Churn'],
            autopct='%1.1f%%',
            colors=['steelblue', 'coral'],
            startangle=90)
    ax1.set_title('Overall Churn Distribution',
                  fontweight='bold')
    st.pyplot(fig1)
    st.write('26.5% of customers churned confirming the '
             'business problem identified at Nexus Telecom.')

with tab2:
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    contract_churn = df_raw.groupby('Contract')['Churn'].apply(
        lambda x: (x == 'Yes').mean() * 100).reset_index()
    contract_churn.columns = ['Contract', 'Churn Rate']
    ax2.bar(contract_churn['Contract'],
            contract_churn['Churn Rate'],
            color=['steelblue', 'coral', 'green'])
    ax2.set_title('Churn Rate by Contract Type',
                  fontweight='bold')
    ax2.set_xlabel('Contract Type')
    ax2.set_ylabel('Churn Rate (%)')
    for i, v in enumerate(contract_churn['Churn Rate']):
        ax2.text(i, v + 0.5, f'{v:.1f}%',
                 ha='center', fontweight='bold')
    st.pyplot(fig2)
    st.write('Month-to-month customers churn at 42.7% compared '
             'to just 2.8% for two year contract customers.')

with tab3:
    fig3, ax3 = plt.subplots(figsize=(7, 4))
    churned_charges = df_raw[
        df_raw['Churn'] == 'Yes']['MonthlyCharges']
    not_churned_charges = df_raw[
        df_raw['Churn'] == 'No']['MonthlyCharges']
    ax3.hist(not_churned_charges, bins=30, alpha=0.7,
             color='steelblue', label='No Churn',
             edgecolor='white')
    ax3.hist(churned_charges, bins=30, alpha=0.7,
             color='coral', label='Churn',
             edgecolor='white')
    ax3.set_title('Monthly Charges Distribution by Churn Status',
                  fontweight='bold')
    ax3.set_xlabel('Monthly Charges ($)')
    ax3.set_ylabel('Number of Customers')
    ax3.legend()
    st.pyplot(fig3)
    st.write('Churned customers paid an average of $74.44 per month '
             'compared to $61.27 for retained customers.')

st.markdown('---')

# ----------------------------------------
# Section 3: Interactive Churn Predictor
# ----------------------------------------
st.header('Customer Churn Risk Predictor')
st.write('Enter a customer profile below to predict their churn risk.')

col1, col2 = st.columns(2)

with col1:
    tenure = st.slider('Tenure (months)', 0, 72, 12)
    monthly_charges = st.slider('Monthly Charges ($)',
                                 18.0, 120.0, 65.0)
    contract = st.selectbox('Contract Type',
                            ['Month-to-month',
                             'One year',
                             'Two year'])

with col2:
    internet = st.selectbox('Internet Service',
                            ['DSL', 'Fiber optic', 'No'])
    senior = st.selectbox('Senior Citizen', ['No', 'Yes'])
    partner = st.selectbox('Has Partner', ['No', 'Yes'])

if st.button('Predict Churn Risk', type='primary'):
    input_df = pd.DataFrame(columns=X.columns)
    input_df.loc[0] = 0
    input_df['tenure'] = tenure
    input_df['MonthlyCharges'] = monthly_charges

    if contract == 'One year':
        if 'Contract_One year' in input_df.columns:
            input_df['Contract_One year'] = 1
    elif contract == 'Two year':
        if 'Contract_Two year' in input_df.columns:
            input_df['Contract_Two year'] = 1

    if internet == 'Fiber optic':
        if 'InternetService_Fiber optic' in input_df.columns:
            input_df['InternetService_Fiber optic'] = 1
    elif internet == 'No':
        if 'InternetService_No' in input_df.columns:
            input_df['InternetService_No'] = 1

    if senior == 'Yes':
        input_df['SeniorCitizen'] = 1

    if partner == 'Yes':
        if 'Partner_Yes' in input_df.columns:
            input_df['Partner_Yes'] = 1

    prob = model.predict_proba(input_df)[0][1] * 100

    st.markdown('---')
    st.subheader('Prediction Results')

    col1, col2 = st.columns(2)
    with col1:
        st.metric('Churn Probability', f'{prob:.1f}%')
    with col2:
        if prob > 60:
            st.error('HIGH RISK')
        elif prob > 40:
            st.warning('MEDIUM RISK')
        else:
            st.success('LOW RISK')

    if prob > 60:
        st.error('Action Required: Contact this customer immediately.')
        st.write('Recommended interventions:')
        st.write('- Offer a contract upgrade discount')
        st.write('- Provide a dedicated customer support call')
        st.write('- Consider a service plan adjustment')
    elif prob > 40:
        st.warning('Monitor this customer closely.')
        st.write('Consider proactive outreach within 30 days.')
    else:
        st.success('No immediate action needed.')
        st.write('Continue standard customer service.')

st.markdown('---')

# ----------------------------------------
# Section 4: Model Performance
# ----------------------------------------
st.header('Model Performance')

col1, col2, col3 = st.columns(3)
with col1:
    st.metric('Accuracy', '82.19%', 'Target: 75%')
with col2:
    st.metric('Precision', '68.71%', 'Target: 65%')
with col3:
    st.metric('Recall', '60.05%', 'Target: 70%')

cm = confusion_matrix(y_test, model.predict(X_test))
fig4, ax4 = plt.subplots(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Churn', 'Churn'],
            yticklabels=['No Churn', 'Churn'])
ax4.set_title('Confusion Matrix', fontweight='bold')
ax4.set_xlabel('Predicted')
ax4.set_ylabel('Actual')
st.pyplot(fig4)

st.markdown('---')
st.caption('Nexus Telecom Customer Churn Prediction | '
           'Built with Python and Scikit-learn')
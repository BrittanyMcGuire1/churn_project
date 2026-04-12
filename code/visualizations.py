import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the raw dataset
df = pd.read_csv('../data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Set visual style
sns.set_style("whitegrid")

# ----------------------------------------
# Chart 1: Overall Churn Distribution
# ----------------------------------------
fig1, ax1 = plt.subplots(figsize=(7, 5))
churn_counts = df['Churn'].value_counts()
ax1.pie(churn_counts,
        labels=['No Churn', 'Churn'],
        autopct='%1.1f%%',
        colors=['steelblue', 'coral'],
        startangle=90)
ax1.set_title('Overall Churn Distribution',
              fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../screenshots/chart1_churn_distribution.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("Chart 1 saved: Overall Churn Distribution")

# ----------------------------------------
# Chart 2: Churn Rate by Contract Type
# ----------------------------------------
fig2, ax2 = plt.subplots(figsize=(8, 5))
contract_churn = df.groupby('Contract')['Churn'].apply(
    lambda x: (x == 'Yes').mean() * 100).reset_index()
contract_churn.columns = ['Contract', 'Churn Rate']
ax2.bar(contract_churn['Contract'],
        contract_churn['Churn Rate'],
        color=['steelblue', 'coral', 'green'])
ax2.set_title('Churn Rate by Contract Type',
              fontsize=14, fontweight='bold')
ax2.set_xlabel('Contract Type', fontsize=12)
ax2.set_ylabel('Churn Rate (%)', fontsize=12)
for i, v in enumerate(contract_churn['Churn Rate']):
    ax2.text(i, v + 0.5, f'{v:.1f}%',
             ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('../screenshots/chart2_churn_by_contract.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("Chart 2 saved: Churn Rate by Contract Type")

# ----------------------------------------
# Chart 3: Monthly Charges Distribution
# ----------------------------------------
fig3, ax3 = plt.subplots(figsize=(8, 5))
churned = df[df['Churn'] == 'Yes']['MonthlyCharges']
not_churned = df[df['Churn'] == 'No']['MonthlyCharges']
ax3.hist(not_churned, bins=30, alpha=0.7,
         color='steelblue', label='No Churn',
         edgecolor='white')
ax3.hist(churned, bins=30, alpha=0.7,
         color='coral', label='Churn',
         edgecolor='white')
ax3.set_title('Monthly Charges Distribution by Churn Status',
              fontsize=14, fontweight='bold')
ax3.set_xlabel('Monthly Charges ($)', fontsize=12)
ax3.set_ylabel('Number of Customers', fontsize=12)
ax3.legend(fontsize=11)
plt.tight_layout()
plt.savefig('../screenshots/chart3_monthly_charges.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("Chart 3 saved: Monthly Charges Distribution")

print()
print("All three visualizations generated and saved to screenshots folder")
# ================================
# STEP 0: Import Libraries
# ================================
import pandas as pd     # pandas for data manipulation
import numpy as np      # numpy for generating random numbers
from sklearn.model_selection import train_test_split  # split train-test
from sklearn.linear_model import LogisticRegression  # Logistic Regression model
from sklearn.ensemble import RandomForestClassifier  # Random Forest model
from sklearn.metrics import roc_auc_score, f1_score   # evaluation metrics
import joblib          # save models

# ================================
# STEP 1: Generate Synthetic Malaysia-style Dataset
# ================================
np.random.seed(42)  # for reproducibility
n_samples = 2000    # number of samples

df = pd.DataFrame({
    # Age (21-65)
    "age": np.random.randint(21, 65, n_samples),

    # Monthly Income (RM3000-10000)
    "monthly_income": np.random.normal(5000, 2000, n_samples).astype(int),

    # Loan Amount (RM1000-50000)
    "loan_amount": np.random.randint(1000, 50000, n_samples),

    # Loan Term in months (12,24,36,48,60)
    "loan_term": np.random.choice([12,24,36,48,60], n_samples),

    # Debt-to-Income Ratio (0-1)
    "dti": np.random.uniform(0, 1, n_samples),

    # Credit History (0-10 past loans)
    "credit_history": np.random.randint(0, 10, n_samples),

    # Number of Dependents (0-5)
    "num_dependents": np.random.randint(0, 6, n_samples),

    # Simulated Borrower Name (Mixed English + Malay)
    "name": ["Ahmad " + str(i) if i%2==0 else "John " + str(i) for i in range(n_samples)],

    # Simulated Job (Diverse list)
    "job": np.random.choice([
        "Software Engineer", "Data Scientist", "Teacher", "Doctor", "Nurse", 
        "Accountant", "Civil Engineer", "Mechanical Engineer", "Graphic Designer", 
        "Marketing Manager", "Sales Executive", "Business Analyst", "Project Manager", 
        "Chef", "Waiter", "Grab Driver", "Delivery Rider", "Police Officer", 
        "Lawyer", "Architect", "Pharmacist", "Dentist", "Electrician", "Plumber", 
        "Mechanic", "Technician", "Factory Worker", "Clerk", "Receptionist", 
        "Human Resources Manager", "Financial Advisor", "Real Estate Agent", 
        "Journalist", "Editor", "Content Creator", "Social Media Manager", 
        "Customer Service Representative", "Operations Manager", "Director", "CEO"
    ], n_samples),

    # Simulated City (All major cities/towns in Malaysia)
    "city": np.random.choice([
        # Kuala Lumpur & Selangor
        "Kuala Lumpur", "Petaling Jaya", "Shah Alam", "Klang", "Subang Jaya", 
        "Puchong", "Ampang", "Cyberjaya", "Putrajaya", "Kajang", "Rawang", "Sepang",
        
        # Penang
        "George Town", "Butterworth", "Bukit Mertajam", "Nibong Tebal", "Bayan Lepas",
        
        # Johor
        "Johor Bahru", "Iskandar Puteri", "Pasir Gudang", "Muar", "Batu Pahat", 
        "Kluang", "Segamat", "Kota Tinggi", "Mersing",
        
        # Perak
        "Ipoh", "Taiping", "Teluk Intan", "Sitiawan", "Kampar", "Kuala Kangsar",
        
        # Kedah
        "Alor Setar", "Sungai Petani", "Kulim", "Langkawi", "Jitra",
        
        # Pahang
        "Kuantan", "Temerloh", "Bentong", "Mentakab", "Raub", "Cameron Highlands",
        
        # Negeri Sembilan
        "Seremban", "Port Dickson", "Nilai", "Bahau",
        
        # Melaka
        "Melaka City", "Alor Gajah", "Jasin",
        
        # Kelantan
        "Kota Bharu", "Pasir Mas", "Tanah Merah", "Gua Musang",
        
        # Terengganu
        "Kuala Terengganu", "Chukai", "Dungun", "Besut",
        
        # Perlis
        "Kangar", "Arau",
        
        # Sabah
        "Kota Kinabalu", "Sandakan", "Tawau", "Lahad Datu", "Keningau", "Semporna",
        
        # Sarawak
        "Kuching", "Miri", "Sibu", "Bintulu", "Sri Aman", "Kapit"
    ], n_samples)
})

# ================================
# Step 1.1: Generate default probability
# We simulate the default label based on loan_amount/income ratio, DTI, credit history, dependents
# ================================
prob_default = (
    0.2 * (df["loan_amount"] / df["monthly_income"]) +       # loan-to-income ratio
    0.3 * df["dti"] +                                        # debt-to-income
    0.2 * (1 - np.exp(-df["credit_history"]/5)) +            # credit history effect
    0.1 * (df["num_dependents"]/5) +                         # dependents effect
    np.random.normal(0,0.05,n_samples)                       # random noise
)

# Convert probability to binary default label
df["default"] = (prob_default > 0.5).astype(int)

# ================================
# STEP 2: Data Cleaning
# ================================
# Ensure monthly income >= RM1000
df["monthly_income"] = df["monthly_income"].apply(lambda x: max(x,1700))

# Check for missing values
print("Missing values in dataset:")
print(df.isnull().sum())

# ================================
# STEP 3: Feature Selection
# ================================
# Select structured features for baseline model
features = ["age", "monthly_income", "loan_amount", "loan_term",
            "dti", "credit_history", "num_dependents"]
X = df[features]
y = df["default"]

# ================================
# STEP 4: Train-Test Split
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================================
# STEP 5: Train Logistic Regression Baseline
# ================================
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

# ================================
# STEP 6: Train Random Forest Baseline
# ================================
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# ================================
# STEP 7: Evaluate Models
# ================================
def evaluate(model, X_test, y_test):
    """
    Evaluate model performance
    Metrics: AUC + F1
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]  # probability of default
    auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    return auc, f1

lr_auc, lr_f1 = evaluate(lr_model, X_test, y_test)
rf_auc, rf_f1 = evaluate(rf_model, X_test, y_test)

print("Logistic Regression -> AUC: {:.3f}, F1: {:.3f}".format(lr_auc, lr_f1))
print("Random Forest       -> AUC: {:.3f}, F1: {:.3f}".format(rf_auc, rf_f1))

# ================================
# STEP 8: Save Cleaned Dataset and Models
# ================================
df.to_csv("cleaned_data.csv", index=False)              # Cleaned dataset
joblib.dump(lr_model, "baseline_model_lr.pkl")          # Logistic Regression model
joblib.dump(rf_model, "baseline_model_rf.pkl")          # Random Forest model

print("✅ All done! CSV and baseline models are saved.")


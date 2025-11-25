# ================================
# STEP 0: Import Libraries / 导入必要库
# ================================
import pandas as pd     # pandas 用于数据处理 / data manipulation
import numpy as np      # numpy 用于生成随机数据 / generate random numbers
from sklearn.model_selection import train_test_split  # 划分训练/测试集 / split train-test
from sklearn.linear_model import LogisticRegression  # Logistic Regression 模型
from sklearn.ensemble import RandomForestClassifier  # Random Forest 模型
from sklearn.metrics import roc_auc_score, f1_score   # 模型评估指标 / evaluation metrics
import joblib          # 保存模型 / save models

# ================================
# STEP 1: Generate Synthetic Malaysia-style Dataset
# Step 1: 生成马来西亚风格的模拟信用数据
# ================================
np.random.seed(42)  # 固定随机种子 / for reproducibility
n_samples = 2000    # 数据条数 / number of samples

df = pd.DataFrame({
    # 年龄 / Age (21-65)
    "age": np.random.randint(21, 65, n_samples),

    # 月收入 / Monthly Income (RM3000-10000)
    "monthly_income": np.random.normal(5000, 2000, n_samples).astype(int),

    # 贷款额度 / Loan Amount (RM1000-50000)
    "loan_amount": np.random.randint(1000, 50000, n_samples),

    # 贷款期限 / Loan Term in months (12,24,36,48,60)
    "loan_term": np.random.choice([12,24,36,48,60], n_samples),

    # 债务收入比 / Debt-to-Income Ratio (0-1)
    "dti": np.random.uniform(0, 1, n_samples),

    # 信用历史次数 / Credit History (0-10 past loans)
    "credit_history": np.random.randint(0, 10, n_samples),

    # 家庭赡养人数 / Number of Dependents (0-5)
    "num_dependents": np.random.randint(0, 6, n_samples),

    # 模拟借款人姓名 / Name (混合英文+马来文)
    "name": ["Ahmad " + str(i) if i%2==0 else "John " + str(i) for i in range(n_samples)],

    # 模拟职业 / Job (Diverse list)
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

    # 模拟城市 / City (All major cities/towns in Malaysia)
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
# Step 1.1: Generate default probability / 生成违约概率
# We simulate the default label based on loan_amount/income ratio, DTI, credit history, dependents
# 根据贷款额度/收入比、债务比率、信用历史、家庭赡养人数生成违约概率
# ================================
prob_default = (
    0.2 * (df["loan_amount"] / df["monthly_income"]) +       # 贷款占收入比例 / loan-to-income ratio
    0.3 * df["dti"] +                                        # 债务比率 / debt-to-income
    0.2 * (1 - np.exp(-df["credit_history"]/5)) +            # 信用历史影响 / credit history effect
    0.1 * (df["num_dependents"]/5) +                         # 家庭负担影响 / dependents effect
    np.random.normal(0,0.05,n_samples)                       # 随机噪声 / random noise
)

# 根据概率生成违约标签 / convert probability to binary default label
df["default"] = (prob_default > 0.5).astype(int)

# ================================
# STEP 2: Data Cleaning / 数据清理
# ================================
# 收入下限处理 / Ensure monthly income >= RM1000
df["monthly_income"] = df["monthly_income"].apply(lambda x: max(x,1700))

# 检查缺失值 / Check for missing values
print("Missing values in dataset / 数据缺失情况：")
print(df.isnull().sum())

# ================================
# STEP 3: Feature Selection / 特征选择
# ================================
# 选择结构化特征用于模型训练 / select structured features for baseline model
features = ["age", "monthly_income", "loan_amount", "loan_term",
            "dti", "credit_history", "num_dependents"]
X = df[features]
y = df["default"]

# ================================
# STEP 4: Train-Test Split / 划分训练集和测试集
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================================
# STEP 5: Train Logistic Regression Baseline / 训练 Logistic Regression 基线模型
# ================================
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

# ================================
# STEP 6: Train Random Forest Baseline / 训练 Random Forest 基线模型
# ================================
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# ================================
# STEP 7: Evaluate Models / 模型评估
# ================================
def evaluate(model, X_test, y_test):
    """
    Evaluate model performance
    评估模型性能: AUC + F1
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]  # 取正类概率 / probability of default
    auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    return auc, f1

lr_auc, lr_f1 = evaluate(lr_model, X_test, y_test)
rf_auc, rf_f1 = evaluate(rf_model, X_test, y_test)

print("Logistic Regression -> AUC: {:.3f}, F1: {:.3f}".format(lr_auc, lr_f1))
print("Random Forest       -> AUC: {:.3f}, F1: {:.3f}".format(rf_auc, rf_f1))

# ================================
# STEP 8: Save Cleaned Dataset and Models / 保存数据集和模型
# ================================
df.to_csv("cleaned_data.csv", index=False)              # Cleaned dataset / 清理后的数据集
joblib.dump(lr_model, "baseline_model_lr.pkl")          # Logistic Regression model / LR模型
joblib.dump(rf_model, "baseline_model_rf.pkl")          # Random Forest model / RF模型

print("✅ All done! CSV and baseline models are saved.")


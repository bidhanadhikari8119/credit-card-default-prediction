# Credit Card Default Prediction
# Business question: Which customers are likely to miss next month's payment,
# so the collections team knows who to call first?

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


# ---------------- 1. LOAD AND CHECK ----------------
df = pd.read_excel(r'default of credit card clients.xls', skiprows=1)
df = df.rename(columns={'default payment next month': 'default'})   # shorter name
df = df.drop(columns='ID')                                           # ID tells us nothing

print('Shape:', df.shape)
print('Missing values:', df.isna().sum().sum())
print('Duplicate rows:', df.duplicated().sum())
print('\nDefault rate:')
print(df['default'].value_counts(normalize=True).round(3))


# ---------------- 2. CLEAN ----------------
# EDUCATION codes 0, 5, 6 are not in the documentation -> move them to 4 (others)
df['EDUCATION'] = df['EDUCATION'].replace({0: 4, 5: 4, 6: 4})

# Drop SEX and MARRIAGE: US fair-lending rules bar using them in credit decisions
df = df.drop(columns=['SEX', 'MARRIAGE'])

# One-hot encode EDUCATION (do this ONCE, after cleaning)
# drop_first drops EDU_1 (graduate school), so that is the baseline - a real category now
df = pd.get_dummies(df, columns=['EDUCATION'], prefix='EDU', drop_first=True, dtype=int)


# ---------------- 3. QUICK EDA: what drives default? ----------------
print('\nDefault rate by last month repayment status (PAY_0):')
print(df.groupby('PAY_0')['default'].agg(['mean', 'count']).round(3))

print('\nDefault rate by credit limit (5 equal-size groups):')
print(df.groupby(pd.qcut(df['LIMIT_BAL'], 5), observed=True)['default'].mean().round(3))

print('\nCorrelation of each feature with default:')
print(df.corr()['default'].drop('default').sort_values(ascending=False).round(3))


# ---------------- 4. TRAIN / TEST SPLIT ----------------
X = df.drop(columns='default')
y = df['default']

# 80% train, 20% test; stratify keeps the same default rate in both
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)


# ---------------- 5. SCALE (for Logistic Regression) ----------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # learn mean/std from TRAIN only
X_test_scaled = scaler.transform(X_test)         # just apply it to TEST (no fit = no leakage)


# ---------------- 6. BASELINE ----------------
# Predicts "no default" for everyone - any real model must beat this
baseline = DummyClassifier(strategy='most_frequent')
baseline.fit(X_train, y_train)
print('\n===== BASELINE =====')
print(classification_report(y_test, baseline.predict(X_test), zero_division=0))


# ---------------- 7. LOGISTIC REGRESSION ----------------
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_scaled, y_train)
log_pred = log_model.predict(X_test_scaled)

print('===== LOGISTIC REGRESSION =====')
print(confusion_matrix(y_test, log_pred))
print(classification_report(y_test, log_pred))


# ---------------- 8. RANDOM FOREST ----------------
rf_model = RandomForestClassifier(random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)            # trees do not need scaled data
rf_pred = rf_model.predict(X_test)

print('===== RANDOM FOREST =====')
print(confusion_matrix(y_test, rf_pred))
print(classification_report(y_test, rf_pred))

importance = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
print('Top 5 features:')
print(importance.head(5).round(3))

importance.head(10).sort_values().plot(kind='barh', title='Top 10 drivers of default')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()


# ---------------- 9. THRESHOLD = BUSINESS DECISION ----------------
# precision = of the customers we call, how many really default
# recall    = of all customers who default, how many we catch
results = []
models = {
    'Logistic': log_model.predict_proba(X_test_scaled)[:, 1],
    'Random Forest': rf_model.predict_proba(X_test)[:, 1],
}
for name, prob in models.items():
    for t in [0.5, 0.3]:
        pred = (prob >= t).astype(int)
        results.append({
            'model': name,
            'threshold': t,
            'precision': round(precision_score(y_test, pred), 2),
            'recall': round(recall_score(y_test, pred), 2),
            'customers_to_call': pred.sum(),
        })

print('\n===== THRESHOLD COMPARISON (test set: 6,000 customers) =====')
print(pd.DataFrame(results))

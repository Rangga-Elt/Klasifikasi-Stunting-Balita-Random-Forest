import io
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from google.colab import files
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, learning_curve

# Upload Dataset
uploaded = files.upload()
file_name = next(iter(uploaded))
if file_name.endswith('.csv'):
    df = pd.read_csv(io.BytesIO(uploaded[file_name]))
elif file_name.endswith('.xlsx'):
    df = pd.read_excel(io.BytesIO(uploaded[file_name]))
else:
    raise ValueError("Format file tidak didukung. Harap unggah file CSV atau XLSX.")

# Validasi Data
assert df['Umur_(bulan)'].between(0, 60).all(), "Umur harus 0-60 bulan!"
assert {'laki-laki', 'perempuan'}.issubset(df['Jenis_Kelamin'].unique()), "Jenis kelamin tidak lengkap!"
assert {'severely stunted', 'stunted', 'normal', 'tinggi'}.issubset(df['Status_Gizi'].unique()), "Kelas target tidak lengkap!"

# Preprocessing 
le_gender = LabelEncoder()
le_target = LabelEncoder()
df['Jenis_Kelamin'] = le_gender.fit_transform(df['Jenis_Kelamin'])  # laki-laki = 0
df['Status_Gizi'] = le_target.fit_transform(df['Status_Gizi'])     # target encoded

scaler = StandardScaler()
df[['Umur_(bulan)', 'Tinggi_Badan_(cm)']] = scaler.fit_transform(df[['Umur_(bulan)', 'Tinggi_Badan_(cm)']])

# Split Data
X = df.drop(columns=['Status_Gizi'])
y = df['Status_Gizi']
stratify_col = df['Umur_(bulan)'].astype(str) + "_" + df['Jenis_Kelamin'].astype(str)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=stratify_col
)

# Hyperparameter Tuning dengan GridSearchCV
param_grid = {
    'n_estimators': [100],
    'max_depth': [12],
    'min_samples_split': [10],
    'min_samples_leaf': [10],
    'max_leaf_nodes': [None],
    'criterion': ['gini', 'entropy', 'log_loss'],
    'class_weight': ['balanced']
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid,
    cv=StratifiedKFold(n_splits=5),
    scoring='accuracy',
    verbose=1,
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

# Final Model
model = grid_search.best_estimator_
print("\nBest Parameters from GridSearchCV:")
print(grid_search.best_params_)

# Evaluasi
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

print("\nAkurasi Training:", accuracy_score(y_train, y_pred_train))
print("Akurasi Testing :", accuracy_score(y_test, y_pred_test))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_test, target_names=le_target.classes_))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_test)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le_target.classes_, yticklabels=le_target.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Prediksi")
plt.ylabel("Aktual")
plt.show()

# Learning Curve
train_sizes, train_scores, test_scores = learning_curve(
    estimator=model,
    X=X_train,
    y=y_train,
    cv=StratifiedKFold(n_splits=5),
    scoring='accuracy',
    n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 5),
    shuffle=True,
    random_state=42
)

train_scores_mean = train_scores.mean(axis=1)
test_scores_mean = test_scores.mean(axis=1)

plt.figure(figsize=(7, 6))
plt.plot(train_sizes, train_scores_mean, 'o-', color="blue", label="Training Accuracy")
plt.plot(train_sizes, test_scores_mean, 'o-', color="red", label="Validation Accuracy")
plt.title("Learning Curve")
plt.xlabel("Jumlah Data Latih")
plt.ylabel("Accuracy")
plt.legend(loc="best")
plt.grid()
plt.show()

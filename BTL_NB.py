import math
from collections import defaultdict
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

class NaiveBayes:

    def __init__(self):
        self.class_probabilities = {}
        self.feature_statistics = {}

    def fit(self, X, y):
        # Số lượng mẫu và số lượng đặc trưng
        n_samples, n_features = X.shape
        
        # Các lớp trong dữ liệu
        classes = np.unique(y)
        
        # Tính xác suất a priori cho mỗi lớp
        for c in classes:
            self.class_probabilities[c] = np.sum(y == c) / n_samples
        
        # Tính toán trung bình và phương sai cho mỗi lớp và đặc trưng
        for c in classes:
            X_c = X[y == c]
            feature_stats = {}
            for feature_idx in range(n_features):
                feature_values = X_c[:, feature_idx]
                mean = np.mean(feature_values)
                var = np.var(feature_values)
                feature_stats[feature_idx] = {'mean': mean, 'var': var}
            self.feature_statistics[c] = feature_stats
    def gaussian_probability(self, x, mean, var):
        # Thêm một giá trị nhỏ để tránh chia cho 0
        epsilon = 1e-10
        var = var + epsilon
        exponent = math.exp(-0.5 * ((x - mean) ** 2) / var)
        return (1 / math.sqrt(2 * math.pi * var)) * exponent
    def predict(self, X):
        predictions = []
        for x in X:
            class_scores = {}
            for c, class_prob in self.class_probabilities.items():
                # Tính xác suất log của lớp
                log_prob = math.log(class_prob)
                # Tính toán xác suất điều kiện cho từng thuộc tính
                for feature_idx, value in enumerate(x):
                    mean = self.feature_statistics[c][feature_idx]['mean']
                    var = self.feature_statistics[c][feature_idx]['var']
                    # Tính xác suất của giá trị với phân phối chuẩn
                    prob = self.gaussian_probability(value, mean, var)
                    log_prob += math.log(prob)
                class_scores[c] = log_prob
            # Chọn lớp có xác suất cao nhất
            predictions.append(max(class_scores, key=class_scores.get))
        return np.array(predictions)

  

df = pd.read_csv("cleaned_data_degree.csv")

mode_value = df['Financial_Stress'].mode()[0]
df['Financial_Stress'] = df['Financial_Stress'].replace('?', mode_value)
def preprocess_data(df):                
    # Xử lý giá trị thiếu cho tất cả các cột
    for column in df.columns:
        if df[column].dtype == 'object':
            mode_value = df[column].mode()[0]
            df[column] = df[column].replace('?', mode_value)
    
    # Tiếp tục xử lý như cũ
    for column in df.columns:
        if df[column].dtype == 'object':
            # Lấy giá trị duy nhất và sắp xếp
            unique_values = sorted(df[column].unique())
            mapping = {value: idx for idx, value in enumerate(unique_values)}
            df[column] = df[column].map(mapping)
        elif df[column].dtype != 'object':
            df[column] = df[column].astype(float)
    return df
df=preprocess_data(df)

X = df.drop('Depression', axis='columns').values
y = df['Depression'].values

def train_test_split_custom(X, y, test_size=0.2, random_state=None):
    np.random.seed(random_state)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    split = int(len(X) * (1 - test_size))
    train_indices, test_indices = indices[:split], indices[split:]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

X_train, X_test, y_train, y_test = train_test_split_custom(X, y, test_size=0.2, random_state=42)
model = NaiveBayes()
model.fit(X_train, y_train)
y_pred =model.predict(X_test)
results_df = pd.DataFrame({
    'True Labels': y_test,
    'Predicted Labels': y_pred
})
results_df.to_csv('predictions.csv', index=False)

accuracy = sum([1 if pred == true else 0 for pred, true in zip(y_pred, y_test)]) / len(y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")
plt.figure(figsize=(8, 6))
sleep_duration=X_test[:,5]
Work_Study=X_test[:,8]
sns.scatterplot(x=sleep_duration,y=Work_Study,hue=y_pred, palette='viridis', s=100, edgecolor='k')
plt.title("Sleep_Duration vs Work/Study Hours with Predicted Classes")
plt.xlabel("Sleep_Duration")
plt.ylabel("Work/Study Hours")
plt.legend(title="Predicted Class")

plt.figure(figsize=(12, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap of Features')
plt.tight_layout()
# Hiển thị biểu đồ
plt.show()

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


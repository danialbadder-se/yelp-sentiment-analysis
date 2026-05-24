import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import json

# =====================
# STEP 1 - READ DATASET
# =====================
print("Reading dataset...")

file_path = r"C:\Users\Dania\OneDrive\Desktop\Open Source software devlpment\yelp_academic_dataset_review.json"

data = []
with open(file_path, encoding="utf-8") as f:
    for i, line in enumerate(f):
        data.append(json.loads(line))
        if i == 99999:  # 100k rows
            break

df = pd.DataFrame(data)
print(f"Dataset loaded! Shape: {df.shape}")

# =====================
# STEP 2 - PREPROCESS
# =====================
print("Preprocessing...")

df = df[['text', 'stars']]
df.dropna(inplace=True)

def label(star):
    if star <= 2:
        return 'Negative'
    elif star == 3:
        return 'Neutral'
    else:
        return 'Positive'

df['sentiment'] = df['stars'].apply(label)
print(df['sentiment'].value_counts())

# =====================
# STEP 3 - MODEL
# =====================
print("Training model...")

X = df['text']
y = df['sentiment']

tfidf = TfidfVectorizer(max_features=5000)
X_vec = tfidf.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# =====================
# STEP 4 - RESULTS
# =====================
print("Generating results...")

acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.2f}")
print(classification_report(y_test, y_pred))

# Graph 1 - Confusion Matrix
plt.figure(figsize=(8,6))
cm = confusion_matrix(y_test, y_pred, labels=['Negative','Neutral','Positive'])
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=['Negative','Neutral','Positive'],
            yticklabels=['Negative','Neutral','Positive'],
            cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('confusion_matrix.png')
plt.show()

# Graph 2 - Sentiment Distribution
plt.figure(figsize=(8,6))
df['sentiment'].value_counts().plot(kind='bar', color=['red','gray','green'])
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.savefig('sentiment_distribution.png')
plt.show()

# Graph 3 - Accuracy Bar Chart
plt.figure(figsize=(6,4))
plt.bar(['Accuracy'], [acc], color='blue')
plt.ylim(0, 1)
plt.title('Model Accuracy')
plt.ylabel('Score')
plt.savefig('accuracy.png')
plt.show()

print("Done! Graphs save ho gayi hain!")
import json
from pythainlp import word_tokenize
from sklearn.model_selection import train_test_split

with open('dataset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

x = []
y = []


for item in data:
    x.append(item['text'])
    y.append(item['label'])

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

print(f"Train: {len(X_train)}")
print(f"Test: {len(X_test)}")

print(f"input len: {len(x)}")
print(f"output len: {len(y)}")

#words = word_tokenize(text, engine="newmm")

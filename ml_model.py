import json
from pythainlp import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

with open('dataset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

x = []
y = []

for item in data:
    x.append(item['text'])
    y.append(item['label'])

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
# print(f"Train: {len(X_train)}")
# print(f"Test: {len(X_test)}")

def thai_tokenizer(text):
    return word_tokenize(text, engine="newmm")

vectorizer = TfidfVectorizer(tokenizer=thai_tokenizer, token_pattern=None)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)

acc = accuracy_score(y_test, y_pred) * 100

print(f"Accuracy: {acc:.2f}%\n")
print(classification_report(y_test, y_pred))

def TestByMe(sentence):
    vec = vectorizer.transform([sentence])
    prediction = model.predict(vec)[0]
    probs = model.predict_proba(vec)[0]
    print(f"Text: {sentence}")
    print(f"-> Model guess: {prediction} with accuracy {max(probs):.2f}%")

TestByMe("พี่ลดเบี้ยให้เหลือสองร้อยบาทนะคะ สนใจไหม")
TestByMe("พูดอะไรนะ สัญญาณไม่ดีเลย ฟังไม่รู้เรื่อง")
TestByMe("เบี้ยประกันตัวนี้ราคาไม่แพงนะจ๊ะ")
TestByMe("วงเงินคุ้มครองชีวิต 5 ล้านบาทถ้วน")

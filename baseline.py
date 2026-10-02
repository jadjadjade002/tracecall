import json
import re #หาตัวเลข

with open('dataset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

def predict(item):
    speaker = item['speaker']
    text = item['text']

    if speaker != 'agent':
        return 'ไม่พบ'

    has_number = bool(re.search(r'\d+', text))

    if (('เบี้ย' in text) or ('ชำระ' in text)) and ('บาท' in text) and has_number:
        return 'พบ'

    return 'ไม่พบ'

correct = 0

for item in data:
    speaker = item['speaker']
    text = item['text']
    actual = item['label']
    predicted = predict(item)
    if predicted == actual:
        correct += 1
    
    print(f"{speaker}: {text}")
    print(f"Actual: {actual} | Predicted: {predicted}\n")

print(f"Correct: {correct}/{len(data)}")
print("Accuracy: ", (correct / len(data)) * 100)
# print("Amount of data: ", len(data))
# print("First data example: ", data[0])
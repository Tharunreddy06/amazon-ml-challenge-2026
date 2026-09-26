from metrics import precision, recall, f1_score

print("Precision:", precision(90, 10))
print("Recall:", recall(90, 20))
print("F1 Score:", f1_score(90, 10, 20))
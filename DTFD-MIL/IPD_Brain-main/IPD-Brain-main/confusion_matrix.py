import pandas as pd
from sklearn.metrics import confusion_matrix

# Load predictions
df = pd.read_csv("D:\\IPD\\IPD-Brain-main\\IPD-Brain-main\\Model--isSaveModel\\abc\\abc\\final_test_predictions.csv")

# Convert to numpy arrays
y_true = df["ground_truth"].values
y_pred = df["predicted_label"].values

# Class names (in correct numeric order)
class_names = ["ASTROCYTOMA", "GLIOBLASTOMA", "OLIGODENDROGLIOMA"]

# Generate confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Create labeled DataFrame
df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)

print("\nConfusion Matrix:")
print(df_cm)

# # Save as CSV
# df_cm.to_csv("confusion_mat.csv")
# print("\nSaved confusion matrix → confusion_mat.csv")

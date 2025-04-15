from data_preprocessing import preprocess_data
from model_training import build_binary_model
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Preprocess data
X_train, X_test, y_train, y_test = preprocess_data("mock_ids_dataset.csv")

# Build model
model = build_binary_model((X_train.shape[1],))

# Train model
print("\nTraining Model...")
history = model.fit(
    X_train,
    y_train,
    epochs=15,
    batch_size=64,
    validation_split=0.1,
    class_weight={0: 1, 1: 4}  # Handle class imbalance
)

# Evaluate
print("\nEvaluation Results:")
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy*100:.2f}%")

# Predictions
y_pred = (model.predict(X_test) > 0.5).astype(int)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Save model
model.save('ids_model.keras')
print("\nModel saved as 'ids_model.keras'")
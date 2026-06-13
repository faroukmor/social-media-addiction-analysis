# Social Media Addiction Classification Pipeline

An end-to-end Machine Learning pipeline using an imbalance-aware `Logistic Regression Pipeline` with automated `StandardScaler` to classify digital addiction levels (`High`, `Medium`, `Low`) with **98% Accuracy** and **0.97 Macro F1-Score**.

### 📊 Model Performance
```text
              precision    recall  f1-score   support

        High       1.00      0.98      0.99      1736
         Low       0.96      1.00      0.98        24
      Medium       0.89      1.00      0.94       240

    accuracy                           0.98      2000
   macro avg       0.95      0.99      0.97      2000
weighted avg       0.99      0.98      0.99      2000
```
🛠️ Key Pipeline Architecture
Outlier Filtering: Automatically drops single-instance target anomalies (Severe) to ensure stable stratified splits.

Feature Engineering: Retains pure behavioral and neurological markers (dopamine_dependency_score, attention_span_score), filtering out non-predictive keys.

Data Leakage Prevention: Integrates StandardScaler directly within the pipeline configuration.

Cost-Sensitive Learning: Utilizes class_weight='balanced' to guarantee 100% Recall on minority classes.

🚀 Interactive CLI Inference
Run the production script dynamically via the terminal for live user evaluation:
```bash
python predict.py

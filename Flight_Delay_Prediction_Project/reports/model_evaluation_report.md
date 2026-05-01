# Flight Delay Prediction - Model Evaluation Report

## Project Objective
Predict whether a flight will be delayed or on time using airline, route, and flight schedule data.

## Dataset Summary
- Source file: `data/raw/Airlines.csv`
- Total rows after cleaning duplicates: 539,383
- Feature columns used: 6
- Target column: `Delay`
- On-time records: 299,119
- Delayed records: 240,264

## Preprocessing
- Dropped columns with more than 45% missing values.
- Converted `FlightDate` to weekday and month if that column exists.
- Encoded categorical route and airline columns with OneHotEncoder.
- Imputed missing numeric values with median and categorical values with most frequent value.
- Dropped ID-like columns: `id`, `Flight` when present.

## Model Performance
| Model               |   Accuracy |   Precision |   Recall |   ROC AUC |
|:--------------------|-----------:|------------:|---------:|----------:|
| Logistic Regression |     0.639  |      0.5943 |   0.5974 |    0.6915 |
| Random Forest       |     0.6473 |      0.6144 |   0.5592 |    0.7007 |

## Best Model
The selected model is **Random Forest**, chosen by the highest ROC AUC score.

## Best Model Confusion Matrix
```text
[[42959, 16865], [21180, 26873]]
```

## Best Model Classification Report
```text
              precision    recall  f1-score   support

           0       0.67      0.72      0.69     59824
           1       0.61      0.56      0.59     48053

    accuracy                           0.65    107877
   macro avg       0.64      0.64      0.64    107877
weighted avg       0.65      0.65      0.65    107877

```

## Generated Files
- Saved model: `models/flight_delay_best_model.pkl`
- Metrics JSON: `reports/metrics.json`
- Confusion matrix: `reports/figures/confusion_matrix.png`
- ROC curve: `reports/figures/roc_curve.png`

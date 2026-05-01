# Flight Delay Prediction Using Machine Learning

This project predicts whether a flight will be delayed or on time using the Kaggle Airlines delay dataset.

## Project Structure

```text
Flight_Delay_Prediction_Project/
├── data/
│   └── raw/
│       └── Airlines.csv
├── models/
│   └── flight_delay_best_model.pkl
├── notebooks/
│   └── Flight_Delay_Prediction.ipynb
├── reports/
│   ├── figures/
│   │   ├── confusion_matrix.png
│   │   └── roc_curve.png
│   ├── metrics.json
│   └── model_evaluation_report.md
├── src/
│   └── train_model.py
├── README.md
└── requirements.txt
```

## How to Run

From this project folder:

```bash
python src/train_model.py
```

The script trains Logistic Regression and Random Forest models, evaluates them with Accuracy, Precision, Recall, ROC AUC, Confusion Matrix, ROC Curve, and Classification Report, then saves the best model as a `.pkl` file.

## Deliverables

- Notebook: `notebooks/Flight_Delay_Prediction.ipynb`
- Trained model: `models/flight_delay_best_model.pkl`
- Evaluation report: `reports/model_evaluation_report.md`
- Metrics: `reports/metrics.json`
- Figures: `reports/figures/`

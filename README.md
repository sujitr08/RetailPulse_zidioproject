# RetailPulse — AI-Powered Customer Analytics & Demand Forecasting

RetailPulse is a Python and Streamlit analytics project for exploring retail transactions, segmenting customers, estimating churn risk, forecasting sales, generating inventory-planning recommendations, and reviewing data/model monitoring results.

## Dashboard

Run the dashboard locally:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL printed by Streamlit in your browser.

## Features

- **General and Executive Overview:** high-level retail performance indicators.
- **Sales & Forecasting:** sales trends, forecast comparisons, and what-if exploration.
- **Customer Segmentation:** RFM-based customer analysis, K-Means, DBSCAN, and PCA artifacts.
- **Churn Risk:** churn risk estimates, evaluation metrics, and feature-importance analysis.
- **Inventory Planning:** demand-based planning recommendations and confidence/spike review.
- **Monitoring & Model Health:** historical drift and model evaluation reports.
- **Data Quality & Exports:** validation results and downloadable outputs.

Verify each feature in the running application before treating it as accepted.

## Project Structure

```text
retailPulse/
├── app.py
├── requirements.txt
├── src/
│   ├── dataProcessing/
│   ├── segmentation/
│   ├── forecasting/
│   ├── churnPrediction/
│   ├── inventoryOptimization/
│   └── monitoring/
├── data/
│   ├── rawData/
│   └── processedData/
├── reports/
│   ├── figures/
│   └── modelReports/
├── notebooks/
├── tests/
└── deployment/
    ├── docker/
    ├── kubernetes/
    ├── airflow/
    └── monitoring/
```

Some folders or components may be empty or incomplete; confirm the actual repository structure before submission.

## Data

The project uses the historical Online Retail workbook. The documented source contains approximately 1.07 million transaction rows and ends in December 2011.

The source data has missing customer identifiers, duplicate records across workbook sheets, returns/cancellations, and extreme transaction quantities. Review the data-quality reports before interpreting results.

Do not commit the original dataset, large intermediate datasets, credentials, or private information to a public repository. Check the dataset's licence and sharing conditions before publishing any data.

## Modeling and Evaluation

The project includes data preparation and quality validation, customer segmentation, Prophet and LSTM forecasting experiments, XGBoost churn prediction, inventory-planning estimates, and monitoring reports.

Model performance must be reported from the saved evaluation artifacts and the final validation run. The current project tracker records unmet forecast and churn targets. Do not describe these targets as achieved.

Inventory recommendations are estimates, not validated savings or purchase orders. Stock-on-hand and unit-cost information are not available in the documented dataset.

## Testing

Run the dashboard smoke tests:

```bash
python -m pytest -q tests/test_dashboard_smoke.py
```

Passing smoke tests do not replace manual testing of every dashboard page, export, and interactive control.

## Reproducibility

1. Use a supported Python version compatible with `requirements.txt`.
2. Install the dependencies.
3. Keep the required local data and generated artifacts in their expected locations.
4. Run the dashboard and tests.
5. Record any differences in metrics caused by library versions or regenerated artifacts.

## Deployment Status

A public deployment must be independently verified before a live URL is listed here. Docker, Kubernetes, workflow automation, and monitoring configurations should only be described as operational after successful execution tests.

## Limitations

- Historical transactions do not represent live retail activity.
- Missing customer IDs may bias customer-level analyses.
- Forecast accuracy is below the documented target.
- Churn validation metrics are below at least one documented target.
- Inventory savings cannot be established without stock, cost, and outcome data.
- Monitoring reports alone do not establish live alerting or automated retraining.

## Author and Project Details

- Author: [Add your name]
- Program: ZIDIO Data Science & Analytics
- Project: RetailPulse
- Repository: [Add the GitHub repository URL]
- Live demo: [Add the verified HTTPS URL]
- Report: [Add the final PDF path or link]
- Demo video: [Add the video URL after recording]

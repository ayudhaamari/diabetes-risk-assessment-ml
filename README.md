# Diabetes Risk Prediction Model 🩺



## 📊 Project Overview

This project develops a machine learning model to predict an individual's risk of developing diabetes. Using the CDC Diabetes Health Indicators dataset, we've created a robust classification model that achieves over 80% recall in identifying potential diabetes cases.

![Diabetes Prediction](/deployment/images/diabetes-ml.jpg)
![Python](https://img.shields.io/badge/Python-3.9-blue.svg) ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Latest-orange.svg) ![Pandas](https://img.shields.io/badge/Pandas-Latest-green.svg) ![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red.svg) ![Plotly](https://img.shields.io/badge/Plotly-Latest-green.svg)

### 🎯 Objective

To build a predictive model that can accurately assess an individual's likelihood of having or developing diabetes, enabling early intervention and better public health strategies.

## 🔍 Key Features

- **Data Source**: CDC Diabetes Health Indicators dataset
- **Primary Metric**: Recall (to minimize false negatives)
- **Target Performance**: 80% recall within six months
- **Models Evaluated**: KNN, SVM, Logistic Regression, Decision Tree, Random Forest, XGBoost
- **Best Performing Model**: Support Vector Machine (SVM)

## 📈 Results

- **Test Set Recall**: 82%
- **Model Performance**: Consistent across train and test sets, indicating good generalization

## 🛠️ Technologies Used

- Python
- Pandas & NumPy
- Scikit-learn
- Streamlit
- Plotly
- Phik (for correlation analysis)

## 📁 Project Structure

```
├── data/
│   └── cdc_diabetes_health_indicators.csv
├── deployment/
│   ├── app.py
│   ├── eda.py
│   ├── prediction.py
│   └── requirements.txt
├── notebooks/
│   └── diabetes-prediction-inference.ipynb
├── images/
│   └── diabetes-ml.jpg
├── README.md
└── tuned_model.pkl
```

## 🚀 Getting Started

1. Clone this repository
2. Install required packages: `pip install -r deployment/requirements.txt`
3. Run the Streamlit app: `streamlit run deployment/app.py`

## 📊 Key Insights

- Age group 60-64 has the highest prevalence of diabetes
- BMI strongly correlates with diabetes risk
- Physical activity and healthy diet are associated with lower diabetes risk

## 🔮 Future Work

- Explore deep learning approaches
- Implement real-time risk assessment system
- Enhance model interpretability

## 👤 Author

Ayudha Amari Hirtranusi

## 🔗 Links

- [Dataset on UCI ML Repository](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators)
- [Hugging Face Space](https://huggingface.co/spaces/amariayudha/Diabetes_Prediction)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🖥️ Web Application

The project includes a web application built with Streamlit. The application offers the following features:

1. **Home Page**: Provides an overview of the project and model performance.
2. **EDA (Exploratory Data Analysis)**: Allows users to explore the dataset through various visualizations.
3. **Prediction**: Enables users to input their health information and receive a diabetes risk assessment.

To run the web application:

```bash
streamlit run deployment/app.py
```

For more details on the application structure, refer to `deployment/app.py`.

## 📊 Exploratory Data Analysis

The EDA module provides interactive visualizations of the dataset. Users can select features for analysis and view distribution plots, box plots, and correlation heatmaps.

For implementation details, see `deployment/eda.py`.

## 🔍 Prediction Module

The prediction module allows users to input their health information and receive a diabetes risk assessment based on the trained model.

For the prediction implementation, refer to `deployment/prediction.py`.

## 🧪 Model Inference

For a detailed example of how to use the trained model for inference, check out the Jupyter notebook `diabetes-prediction-inference.ipynb`. This notebook demonstrates how to load the model and make predictions on new data.
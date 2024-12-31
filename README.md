# Bank-Marketing-Prediction-
# Bank Marketing Subscription Prediction

This repository contains the implementation and analysis of a machine learning project aimed at predicting whether a customer will subscribe to a term deposit based on the *Bank Marketing Dataset*. The dataset, obtained from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing), provides information about past marketing campaigns.

## Project Objective

The primary goal of this project is to classify customer responses to marketing campaigns using machine learning models, specifically:
- *Random Forest Classifier*
- *Neural Network*

## Key Features

1. *Preprocessing Steps:*
   - Data cleaning, including handling missing values and outliers.
   - Feature engineering (e.g., one-hot encoding, interaction terms, log transformations).
   - Balancing the dataset using *SMOTE-Tomek* to address class imbalance.
   - Standardizing numerical features for consistent scaling.

2. *Visualization Techniques:*
   - Heatmaps for correlation analysis.
   - Boxplots and histograms for feature distribution.
   - Count plots for categorical feature analysis.

3. *Model Development:*
   - Training and evaluation of a *Random Forest Classifier*.
   - Designing a *Neural Network* with dropout layers for regularization.

4. *Evaluation Metrics:*
   - Accuracy, Precision, Recall, F1-Score.
   - ROC-AUC scores.
   - Confusion matrices and ROC curves for comparison.

5. *Experimental Results:*
   - Random Forest achieved *94% accuracy* and *ROC-AUC 0.98*.
   - Neural Network achieved *90% accuracy* and *ROC-AUC 0.96*.

6. *Feature Importance:*
   - Visualization of significant features using Random Forest.

## Repository Contents

- *Bank_Marketing_Prediction.ipynb*: The Jupyter notebook containing the complete code implementation, from data preprocessing to model evaluation.
- *Data Files*:
  - bank-additional-full.csv (Primary dataset used).
  - bank-full.csv (Alternative dataset considered).
- *Visualizations*: Plots generated for analysis and evaluation.
- *Documentation*: Detailed description of preprocessing steps, methodology, and results.

## Getting Started

### Prerequisites
- Python 3.8 or later
- Required Python libraries:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  - imbalanced-learn
  - tensorflow

### Installation
1. Clone the repository:
   bash
   git clone https://github.com/Shaithra2107/Bank-Marketing-Prediction.git
   
2. Navigate to the project directory:
   bash
   cd Bank-Marketing-Prediction
   
3. Install the required packages:
   bash
   pip install -r requirements.txt
   

### Usage
1. Open the Jupyter Notebook:
   bash
   jupyter notebook Bank_Marketing_Prediction.ipynb
   
2. Follow the notebook to preprocess the data, train the models, and evaluate their performance.

## Results

| Metric             | Random Forest | Neural Network |
|---------------------|---------------|----------------|
| Accuracy           | 94%           | 90%            |
| ROC-AUC            | 0.98          | 0.96           |
| Precision (Class 1)| 94%           | 92%            |
| Recall (Class 1)   | 94%           | 87%            |

## Limitations

- *Class Imbalance*: Despite using SMOTE-Tomek, real-world scenarios might still pose challenges for the minority class.
- *Dependence on Certain Features*: The duration column, which was removed to avoid data leakage, highlights the need for alternative predictive features.

## Future Enhancements

1. Explore ensemble methods to combine predictions from multiple models.
2. Incorporate real-time data streams for dynamic predictions.
3. Investigate advanced balancing techniques and additional economic features.

## References

1. [Scikit-learn Documentation](https://scikit-learn.org/)
2. Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5–32. DOI: 10.1023/A:1010933404324
3. Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.

## Acknowledgments
This project is part of the *Machine Learning Module (CM2604)* coursework at *Robert Gordon University* and *IIT*.

---

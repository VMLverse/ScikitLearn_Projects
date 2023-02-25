# Scikit Learn Experiments

## Description

This experiments uses scikit learn package to run various algoirhtms to predict the target variable based on a set of input features. The following classification algoritms are used for comparision:
- Decision Tree
- Decision Tree + Adaboost
- Artificial Neural Network (MLPClassifier)
- SVM
- KNN


## Dataset
There are two datasets used in this experiment: 
- the first dataset is [Pima Indian Diabetes](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) which contains 768 observations, with 8 input features and 1 output feature. The attributes are quantitative real numbers and the objective is find whether a patient has diabetes or not.
- the second dataset is [Fetal Health Classification](https://www.kaggle.com/andrewmvd/fetal-health-classification) containing 2126 instances, with 21 input features and the objective is to predict one of three fetal health states.

The features are all quantitative containing both real & integer values.
The diabetes dataset is a binary classification problem whereas the fetal health is a multiclass problem.
The fetal health dataset contains 94% more observations than the diabetes data which can help observe the effect of training size on algorithms.
Both the datasets are unbalanced with the diabetes dataset containing 65% non-diabetic & 35% diabetic instances whereas the other dataset has fetuses in 78% Normal, 14% Suspect & 8% Pathological states.

## Files

- `DecisionTree_Diabetes.ipynb`: Jupyter notebook containing the code for Decision Tree experiment on Diabetes dataset.
- `DecisionTree_FetalHealth.ipynb`: Jupyter notebook containing the code for Decision Tree experiment on Fetal Health dataset.
- `ANN_Diabetes.ipynb`: Jupyter notebook containing the code for MLP Classifier experiment on Diabetes dataset.
- `ANN_FetalHealth.ipynb`: Jupyter notebook containing the code for MLP Classifier experiment on Fetal Health dataset.
- `Boosting_Diabetes.ipynb`: Jupyter notebook containing the code for Adaboost Classifier experiment on Diabetes dataset.
- `Boosting_FHealth.ipynb`: Jupyter notebook containing the code for Adaboost Classifier experiment on Fetal Health dataset.
- `SVM_Diabetes.ipynb`: Jupyter notebook containing the code for SVM Classifier experiment on Diabetes dataset.
- `SVM_FHealth.ipynb`: Jupyter notebook containing the code for SVM Classifier experiment on Fetal Health dataset.
- `KNN_Diabetes.ipynb`: Jupyter notebook containing the code for K-Nearest Neighbor experiment on Diabetes dataset.
- `KNN_FetalHealth.ipynb`: Jupyter notebook containing the code for K-Nearest Neighbor experiment on Fetal Health dataset.
- `diabetes.csv`: CSV file containing the raw diabetes data used in the experiments.
- `fetal_health.csv`: CSV file containing the raw fetal_health data used in the experiments.
- `README.md`: This file, containing information about the experiment.

## Installation

To run this experiment, you will need to have the following libraries installed:

- `pandas`
- `numpy`
- `scikit-learn`

To install these libraries, run the following command:
```bash
pip install pandas numpy scikit-learn
```

## Usage

To run the experiment, open each .ipynb notebook and run each cell in order. The notebook will load the data, preprocess it, train the decision tree model, and evaluate its performance. The notebook also contains comments explaining each step of the code.

## Results

| Dataset         | Algorithm    | F1 score | Precision | Recall |
|-----------------|--------------|----------|-----------|--------|
| Diabetes        | Decision Tree| 0.79     | 0.67      | 0.78   |
| Diabetes        | DT+ AdaBoost | 0.78     | 0.70      | 0.63   |
| Diabetes        | MLPClassifier| 0.70     | 0.59      | 0.52   |
| Diabetes        | SVM          | 0.69     | 0.59      | 0.46   |
| Diabetes        | KNN          | 0.78     | 0.77      | 0.56   |
| Fetal Health    | Logistic Reg | 0.92     | 0.93      | 0.90   |
| Fetal Health    | Decision Tree| 0.90     | 0.90      | 0.91   |
| Fetal Health    | DT+ AdaBoost | 0.89     | 0.89      | 0.88   |
| Fetal Health    | MLPClassifier| 0.91     | 0.91      | 0.91   |
| Fetal Health    | SVM          | 0.92     | 0.92      | 0.92   |
| Fetal Health    | KNN          | 0.90     | 0.90      | 0.90   |


## Conclusion

Even Though Decision trees are intuitive & easy to tune, it is sensitive to outliers. On the contrary, Neural Networks (MLPClassifiers) have too many hyperparameters and difficult to tune.  Although SVMs are versatile with various kernels, training complexity is exponential and is not scalable for large datasets. Similarly Boosting (Adaboost) is not scalable as learning among predictors in the ensemble is sequential. While KNN scales linearly with number of samples, it is sensitive to noise.

Since both datasets are medical applications, the classifier's recall score of the positive scenario is more important as a sick patient (true positive) should not be classified as healthy (false negative) and ignored treatment. Hence a high recall score on out-of-sample data determines if the classifier is 'Best' for our use-case. By the above table, Decision tree has the best recall(0.78) for Diabetes data. Although, SVM has highest recall for fetal health(0.92), the fit time for Decision tree is 114\% lower than SVM and has similar recall(0.91). Hence Decision Tree is the Best classifier for both our datasets.

## Credits

This experiment was created by [Vimal Venugopal](vimalkumar.engr@gmail.com). 

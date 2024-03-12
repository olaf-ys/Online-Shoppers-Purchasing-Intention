# Online-Shoppers-Purchasing-Intention
-- Yuanshan Zhang, Simiao Ye, Mengxin Zhao, Jiayun Liu

In this project, we applied a variety of ML classifiers to a binary classification problem - predicting whether a customer will purchase or not based on his/her page viewing data collected during an online shopping session. Models we used include: Logistic Regression, Random Forest, SVM, XG Boost, and MLP, and our goal is to recommend the model with
the best ability to distinguish between the positive and negative classes to minimize the cost of FPs (False Positives). Therefore, we used the roc_auc (receiver operating characteristic_area under curve) score as our evaluation metric.

## What I did
**1. EDA**\
First, I categorized categorical and numerical features. Second, I visualized the target 'Revenue' and determined the baseline accuracy to be 0.844 (84.4% of the samples belong to negative class). Then I visualized categorical features using bar plots and numerical features using pairplot and heatmap.

From the visualizations, I identified data issues such as class imbalance, multicollinearity, and the skewness of numerical features and resolved these data issues correspondingly in the preprocessing pipeline I built.

**2. Building Pipeline**\
To ensure a simple, coherent, and repeatable data preprocessing and feature engineering process, I streamlined the workflow of feature engineering (i.e. scaling, transformation, and encoding), dimensionality reduction, and oversampling using the scikit-learn pipeline. 

<img src="images/sklearn_pipeline.png" alt="示例图片" width="755" height="332">

**3. MLP**

<img src="images/MLP.png" alt="示例图片" width="485" height="371">

Finally, I designed the MLP structure by searching over 30 combinations of hyper-parameters towards best roc_auc using BayesianOptimization:

- batch size: [32, 64, 128]
- number of layers: [2,3]
- number of neurons on each layer: 10 to 100, step=20
- activation: ['relu', 'tanh']
- dropout rate: 0.1 to 0.5, step=0.1

After fine-tuning, MLP gives the highest validation roc_auc of 0.934 among all the models we built and fine-tuned:

| models | roc_auc (fine-tuned) |
|-------|-------|
| Logistic Regression | 0.915 |
| Random Forest | 0.926 |
| SVM | 0.920 |
| XGBoost | 0.928 |
| MLP | 0.934 |

## Conclusion
| models | accuracy | recall | f1_score | roc_auc | roc_auc (fine-tuned) |
|-------|-------|-------|-------|-------|-------|
| Logistic Regression | 0.892 | 0.559 | 0.622 | 0.915 | NA |
| Random Forest | 0.894 | 0.535 | 0.619 | 0.920 | 0.926 |
| SVM | 0.863 | 0.836 | 0.661 | 0.916 | 0.920 |
| XGBoost | 0.891 | 0.580 | 0.628 | 0.913 | 0.928 |
| MLP | NA | NA | NA | NA | 0.934 |

**Note: it is computationally exhuasting to compute cross validation metrics for MLP, so I only calculated fine-tuned roc_auc.*

While the MLP model achieves the highest fine-tuned roc_auc score and fulfills our goal, the lack of comprehensive cross-validation data (such as accuracy, recall, and f1-score) makes it challenging to fully assess its performance compared to other models. Considering the class imbalance issue and the importance of a balanced assessment across various metrics, the SVM model stands out for its significantly higher recall and f1-score, despite having a slightly lower roc_auc compared to Random Forest, XGBoost, and MLP. Given its well-balanced performance and the importance of recall and f1-score in the context of our imbalanced dataset, I recommend the SVM model to our stakeholders as an alternative. However, the high roc_auc score of the MLP model should not be overlooked, and with further evaluation, it could also be a strong candidate.

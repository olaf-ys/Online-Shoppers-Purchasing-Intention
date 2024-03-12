# Online-Shoppers-Purchasing-Intention
-- Yuanshan Zhang, Simiao Ye, Mengxin Zhao, Jiayun Liu

In this project, we applied a variety of ML classifiers to a binary classification problem - predicting whether a customer will purchase or not based on his/her page viewing data collected during an online shopping session. Models we used include: Logistic Regression, Random Forest, SVM, XG Boost, and MLP, and our goal is to pick the model with highest ROC_AUC (Receiver Operating Characteristic_Area Under Curve) score (i.e. the model that can best distinguishes bewteen the positive class and the negative class)

# What I did
**1. EDA**\
First, I categorized categorical and numerical features. Then I visualized the target 'Revenue' and categorical features using bar plots and numerical features using pairplot and heatmap.

From the visualizations, I observed data issues such as class imbalance, multicolinearity, and the skewness of numerical features, and resolved these data issues in the preprocessing pipeline I built.

**2. Preprocessing Pipeline**\
To assure a simple, coherent, and repeatable data preprocessing and feature engineering process, I streamlined the workflow of feature engineering (i.e. scaling, transformation, and encoding), dimensionality reduction, and oversampling using the scikit-learn pipeline. 

**3. MLP**\
Finally, I built and fine-tuned an Multi-Layer-Perception using TensorFlow and integrated it into the pipeline I built.

# Data Attribution

Data attribution is the study of the relation between data and ML predictions. In downstream applications, data attribution methods can help interpret and compare models; curate datasets; and assess learning algorithm stability. (from https://ml-data-tutorial.org/)

## Links

https://ml-data-tutorial.org/

https://ml-data-tutorial.org/assets/DataTutorialICML2024.pdf

https://transformer-circuits.pub/2024/scaling-monosemanticity/

## People

https://andrewilyas.com/ [twitter](https://x.com/andrew_ilyas) [google scholar](https://scholar.google.ca/citations?user=Dtw3YBoAAAAJ&hl=en)

https://www.cs.cornell.edu/~jsteinhardt/

## Papers

Datamodels: Predicting Predictions from Training Data: https://arxiv.org/pdf/2202.00622

TRAK: Attributing Model Behavior at Scale: https://arxiv.org/pdf/2303.14186

Towards Tracing Factual Knowledge in Language Models Back to the
Training Data: https://arxiv.org/pdf/2205.11482

A Bayesian Approach To Analysing Training Data Attribution In Deep Learning: https://arxiv.org/abs/2305.19765

Decomposing and Editing Predictions by Modeling Model Computation: https://arxiv.org/abs/2404.11534

---

## ChatGPT Overview of Data Attribution

https://chatgpt.com/share/7432067e-d731-4c30-a942-2a9baf1166f2

Data attribution is an important aspect of machine learning that focuses on understanding how different pieces of data influence the predictions made by models. This field is especially crucial for interpreting models, curating datasets, and assessing the stability of learning algorithms. Here's an overview of these topics:

### Methods and Techniques for Data Attribution

Data attribution methods aim to quantify the contribution of each data point or subset of data to the performance of a machine learning model. These methods help understand the influence of training data on the model's predictions and are pivotal for debugging models, improving data quality, and ensuring robust training.

1. **Influence Functions**:

   - Influence functions approximate how a model's predictions would change if a particular training example were modified or removed. This method involves computing the gradient of the loss with respect to the model's parameters and the training data.
   - By examining these gradients, influence functions can provide insights into which training examples have the most impact on the model's predictions.
   - This approach is particularly useful for identifying outliers or mislabeled data that disproportionately affect model performance.

2. **Shapley Values**:

   - Shapley values, derived from cooperative game theory, provide a way to fairly allocate the "payout" (or contribution) of each data point based on its marginal contribution to all possible subsets of the data.
   - In the context of data attribution, Shapley values assess the contribution of each data point to the model's predictions. They offer a rigorous method for data valuation by considering all possible combinations of data points.
   - Calculating Shapley values can be computationally intensive, especially for large datasets, but they provide a comprehensive view of data importance.

3. **Leave-One-Out (LOO) Analysis**:

   - LOO involves systematically removing one data point at a time from the training set and retraining the model to observe the change in performance.
   - This method directly measures the impact of each data point on the model’s overall performance but can be computationally expensive for large datasets or complex models.
   - LOO analysis can highlight which data points are critical for maintaining model accuracy and stability.

4. **Gradient-based Attribution Methods**:
   - These methods, similar to influence functions, use gradients to understand the impact of small perturbations in the data on model predictions.
   - Techniques such as Integrated Gradients or Layer-wise Relevance Propagation (LRP) are popular in interpreting how individual data points contribute to specific model decisions, particularly in deep learning models.

### Assessing Learning Algorithm Stability

Assessing the stability of learning algorithms is crucial for understanding how sensitive a model is to variations in the training data. Stable algorithms are less likely to produce drastically different models when the training data changes slightly.

1. **Resampling Techniques**:

   - Methods like cross-validation, bootstrapping, and jackknife resampling are used to assess how stable a model's performance is across different subsets of the data.
   - By repeatedly training the model on different data splits, one can measure the variability in model performance metrics (e.g., accuracy, precision, recall) to gauge stability.

2. **Noise Sensitivity Analysis**:

   - This involves adding small amounts of noise to the training data and observing how the model's performance and predictions change.
   - A model that is highly sensitive to noise in the data is considered unstable and potentially overfitted to the specific training set.
   - Noise sensitivity analysis helps identify robust models that generalize well to new, unseen data.

3. **Stability Metrics**:

   - Metrics such as the _stability index_ or _influence ranking stability_ are designed to quantify how much model predictions or feature importance rankings change when the training data is perturbed.
   - These metrics provide a quantitative measure of a model’s sensitivity to data variations, helping in model selection and evaluation.

4. **Regularization Techniques**:
   - Regularization methods like L1, L2 regularization, or dropout (in neural networks) are used to enforce stability by penalizing overly complex models that might be sensitive to minor data changes.
   - By constraining the model complexity, regularization helps in achieving models that are not only accurate but also stable.

### Dataset Curation for Model Training

Curating datasets is a vital step in building robust machine learning models. The quality and representativeness of the training data directly impact model performance and generalization ability.

1. **Data Cleaning**:

   - Removing duplicates, handling missing values, correcting mislabeled examples, and dealing with outliers are fundamental to ensuring data quality.
   - Clean datasets help in training models that perform reliably and reduce the chances of overfitting to noise or irrelevant features.

2. **Data Augmentation**:

   - Data augmentation techniques create variations of the existing data to increase the size and diversity of the training set without collecting new data. This is commonly used in image, audio, and text data.
   - Techniques like rotations, flips, noise addition, and synonyms replacement in text help models generalize better by exposing them to a wider variety of input patterns.

3. **Active Learning**:

   - Active learning involves iteratively selecting the most informative data points to be labeled and added to the training set, based on the current state of the model.
   - This method helps in efficiently curating datasets by focusing labeling efforts on examples that are most likely to improve model performance.

4. **Data Diversity and Representativeness**:

   - Ensuring the training dataset is diverse and representative of the target domain is crucial for developing models that generalize well.
   - Techniques like stratified sampling, clustering, or ensuring demographic diversity are employed to maintain a balanced dataset that captures the true underlying distribution of the data.

5. **Feature Engineering and Selection**:

   - Selecting the most relevant features and engineering new ones can significantly improve model performance and reduce the dataset's dimensionality, which enhances stability.
   - Techniques like correlation analysis, recursive feature elimination, or using domain knowledge help in curating datasets that are optimal for training robust models.

6. **Automated Data Labeling and Annotation**:
   - Leveraging semi-supervised learning techniques, transfer learning, or weak supervision can assist in labeling data at scale when human annotation is expensive or impractical.
   - Automated labeling methods can be refined with human oversight to ensure high-quality training datasets.

### Summary

Data attribution methods like influence functions, Shapley values, and LOO analysis help understand the importance of data points in model training. Assessing learning algorithm stability is essential for developing models that are not overly sensitive to data changes, using techniques like resampling, noise sensitivity analysis, and regularization. Finally, effective dataset curation strategies such as data cleaning, augmentation, active learning, and ensuring diversity are fundamental to training robust and generalizable machine learning models.

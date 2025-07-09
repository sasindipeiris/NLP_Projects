# Overview

This notebook aims to classify whether tweets are about real disasters or not using NLP techniques and a deep learning model built with Keras and TensorFlow. The dataset contains 10,000+ hand-labeled tweets.

# Libraries and dependencies

| Library            | Purpose / Usage                                                                                 |
|--------------------|--------------------------------------------------------------------------------------------------|
| **keras-core**     | - Core deep learning framework used to build and train the model. <br> - Provides modular layers (`Dense`, `Sequential`, etc.). <br> - Supports multiple backends (TensorFlow used here). <br> - Enables performant, portable ML model development. |
| **keras-nlp**      | - NLP-specific tools integrated with Keras Core. <br> - Used for tokenizing text (BERT tokenizer). <br> - Includes `TokenAndPositionEmbedding` for sequence modeling. <br> - Simplifies end-to-end text classification pipelines. |
| **pandas**         | - Loads and handles CSV datasets. <br> - Used for data exploration and preprocessing.            |
| **numpy**          | - Supports numerical computations and array manipulation. <br> - Helps with model input formatting. |
| **matplotlib**     | - Plots training metrics and confusion matrix. <br> - Useful for simple visual diagnostics.       |
| **seaborn**        | - Enhances Matplotlib visuals with styled charts. <br> - Used to draw the heatmap of confusion matrix. |
| **sklearn.metrics**| - Computes evaluation metrics like confusion matrix. <br> - `ConfusionMatrixDisplay` visualizes model accuracy. |
| **os**             | - Configures the backend (`KERAS_BACKEND = 'tensorflow'`). <br> - Controls environment settings.  |

# Dataset

The dataset is read from CSV files:

Training set: Contains tweet texts and their corresponding labels (1 for disaster-related, 0 for not).

Test set: Includes unlabeled tweets for potential evaluation or submission.

Initial dataset stats:

Number of training samples: ~7,613

Memory usage and shape of datasets are printed for understanding data volume.

# EDA

As part of the Exploratory Data Analysis (EDA), the number of characters in each tweet was calculated to understand the length distribution of text entries. This was done by applying a character count function to the "text" column in both the training and test datasets, storing the results in a new "length" column.

The .describe() method was then used to generate summary statistics for tweet lengths, including the mean, standard deviation, minimum, maximum, and key percentiles. These statistics help assess how long typical tweets are, identify potential outliers (e.g., extremely short or long tweets), and inform decisions about truncation or padding length during preprocessing.

# Data Preprocessing

The preprocessing pipeline prepares raw tweet text for input into a DistilBERT-based classifier. Each step serves a specific purpose to ensure compatibility with the model and optimal training performance:

## 🔄 Preprocessing Stages

| Step | Description | Reasoning |
|------|-------------|-----------|
| **1. Data Splitting** | The dataset is split into training and validation sets using `train_test_split`. `X` contains tweet texts and `y` contains binary labels. 80% of data is used for training, 20% for validation. | Ensures unbiased model evaluation and generalization testing. `random_state=42` guarantees reproducibility. |
| **2. DistilBERT Preset Selection** | Selected `"distil_bert_base_en_uncased"` as the model preset. | DistilBERT is lightweight and fast, while retaining ~95% of BERT’s accuracy. Ideal for real-time NLP tasks like tweet classification. |
| **3. Preprocessor Initialization** | Initialized `DistilBertPreprocessor` with a sequence length of 160 tokens. It performs tokenization, lowercasing, truncation, and padding. | Matches input format expected by DistilBERT. Ensures consistency in token representation and input shape. |
| **4. Model Initialization** | A `DistilBertClassifier` is created with `num_classes=2` and the preprocessor passed in. | Provides an end-to-end text classification pipeline. Simplifies integration and reduces preprocessing overhead. |
| **5. Batching & Training Config** | Set `BATCH_SIZE = 32`, `EPOCHS = 2`, and computed `STEPS_PER_EPOCH` from training data. Defined `AUTO` for data pipeline optimization. | Optimizes training performance and iteration speed. Smaller epoch count allows quick experimentation. |
| **6. Test Data Prep** | Extracted `X_test` (unlabeled tweet texts) for future prediction use. | Keeps raw test data ready for inference through the same preprocessor-model pipeline. |

# Model and training

The final model used for disaster tweet classification is a **DistilBERT-based neural network**, built with `keras-nlp` and compiled using the `keras-core` framework. The model is designed for **binary text classification**, where tweets are classified as either disaster-related or not.

**Model Configuration:**

* **Base Model**: `DistilBertClassifier` from `keras_nlp.models`, using the `"distil_bert_base_en_uncased"` preset.
* **Input Preprocessing**: Handled automatically by `DistilBertPreprocessor` with a maximum sequence length of 160 tokens.
* **Number of Classes**: 2 (disaster or not).
* **Loss Function**: `SparseCategoricalCrossentropy` with `from_logits=True`, suitable for integer-encoded labels and raw model outputs.
* **Optimizer**: `Adam`, imported from `keras_core.optimizers`.
* **Metric**: Accuracy, to monitor both training and validation performance.

**Training Setup:**

* **Batch Size**: 32
* **Epochs**: 2 (for fast initial experimentation)
* **Training/Validation Split**: 80/20 using `train_test_split`
* **Steps per Epoch**: Computed from training set size and batch size to ensure efficient batching.

The model is compiled and trained using the `classifier.fit()` method, which includes:

* Raw tweet texts (`X_train`) and labels (`y_train`)
* Batch size and number of epochs
* Validation data (`X_val`, `y_val`) for real-time monitoring

The use of pretrained DistilBERT significantly boosts language understanding while keeping training time low. With this setup, the model quickly learns to generalize on short texts like tweets.

# Model Evaluation

**Confusion Matrix**
The confusion matrix visualizes the model's classification results by comparing true labels (y_true) with predicted labels (y_pred).

It breaks down the predictions into four categories:

True Positives (TP): Correctly predicted disaster tweets.

True Negatives (TN): Correctly predicted non-disaster tweets.

False Positives (FP): Tweets incorrectly classified as disasters.

False Negatives (FN): Tweets incorrectly classified as non-disasters.

The ConfusionMatrixDisplay from sklearn.metrics is used to plot these results with a blue color map for clarity.

**F1 Score**

The F1 score is computed alongside the confusion matrix to provide a more balanced measure of model performance, particularly useful in datasets with class imbalance. It combines both precision and recall into a single metric by calculating:
F1 = 2 × TP / (2 × TP + FP + FN)
This ensures that both false positives and false negatives are taken into account. A high F1 score indicates that the model is effectively balancing precision (correctness of positive predictions) and recall (coverage of actual positives). By reporting the F1 score on both training and validation sets, we get a clearer picture of how well the model generalizes and handles difficult cases.




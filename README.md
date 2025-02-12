# Twitter Sentiment Analysis Project

  ## **Sentiment Prediction**
![Image](https://github.com/user-attachments/assets/dfb1cf50-b589-40ea-b8cc-890488998586)


# Project Overview
This project focuses on performing sentiment analysis on a dataset of tweets. 

The goal is to classify tweets as positive or negative based on their content. 

# The Colab notebook follows a structured pipeline;
  - **Data Cleaning**
  - **Exploratory Data Analysis (EDA)**
  - **Feature Engineering**
  - **Model Training**\n
        - Traditional Model(LogisticRegression)
        - Long Short-Term Memory(LSTM)
        - Transfer Model (BERT)


# Dataset
The dataset is Downloaded and stored in google drive for easy access.
This dataset contains the following features,
  - **id:** Unique identifier for each tweet.
  - **label:** Sentiment label (0 for negative, 1 for positive).
  - **tweet:** The text content of the tweet.

---
## Preprocessing Steps
The dataset undergoes several preprocessing steps to prepare it for analysis and modeling this includes removing special characters, converting all text to lowercase, and removing stop.

1. **Removing User Handles:** User handles (e.g., @user) are removed from the tweets.

2. **Removing Special Characters, Punctuations, and Numbers:** Non-alphabetic characters are replaced with spaces.

3. **Removing Short Words:** Words with fewer than 3 characters are removed.

4. **Tokenization:** The tweets are split into individual words.

5. **Stemming:** Words are reduced to their root form using the Porter Stemmer.


----

## Exploratory Data Analysis (EDA)

**Word Cloud**

A word cloud is generated to visualize the most frequent words in the dataset. This helps in understanding the common themes and sentiments expressed in the tweets.

**Correlation Matrix**

A heatmap of the correlation matrix is plotted to understand the relationships between the numeric columns in the dataset.

**Missing Values Handling**

The dataset is checked for missing values but there were no missing values seen.

**Hashtag Analysis**

Hashtags are extracted from both positive and negative tweets, and the most frequent hashtags are visualized using bar plots.


## Model Implementation

The following models are implemented and compared for their performance on the sentiment analysis task:
Three Methods were used,

## Traditional Model (Logistic Regression)

A Logistic Regression model is trained using TF-IDF vectorization.
The TF-IDF vectorizer converts the text data into a matrix of TF-IDF features, which is then used to train the Logistic Regression model.
The model is evaluated using accuracy and F1 score.

  - **Accuracy Score:** 0.9387

  - **F1 Score:** 0.6316

## Deep Learning Model (LSTM)
A deep learning model using LSTM (Long Short-Term Memory) layers is implemented. 
The model architecture includes:

- **Embedding Layer:** Converts words into dense vectors of fixed size.

- **LSTM Layers**: Two LSTM layers with 128 and 64 units respectively, to capture sequential information in the text.

- **Batch Normalization:** Normalizes the outputs of the LSTM layers.

- **Dropout:** Regularization technique to prevent overfitting.

- **Dense Layer:** Output layer with a sigmoid activation function for binary classification.


## Fine-Tuned Model (LSTM with Hyperparameter Tuning)
The deep learning model is fine-tuned with different hyperparameters to improve performance. 
The fine-tuned model includes:

**Embedding Layer:** Increased input dimension to 10,000 and output dimension to 256.

**LSTM Layers:** Two LSTM layers with 256 and 128 units respectively, and added recurrent dropout to prevent overfitting.

**Dense Layers:** Additional dense layer with 64 units and ReLU activation.

**Dropout:** Increased dropout rate to 0.3 for better regularization.


## Transformer Model (BERT)
A BERT-based model is implemented for sentiment analysis.
BERT (Bidirectional Encoder Representations from Transformers) is a powerful transformer-based model that captures context from both directions in a text. The model is fine-tuned on the Twitter sentiment dataset.

**Accuracy Score:** 0.9301

## Evaluation Metrics

- LSTM has the best overall performance among the three models, with the highest accuracy (95.52%) and a decent F1 score (0.6728).
- It strikes a good balance between performance and complexity, making it a strong choice for sentiment analysis tasks.

- BERT has the potential to outperform LSTM, but in this project, it was not fine-tuned extensively, so its performance is slightly lower. 
- With more training and fine-tuning, BERT could achieve better results.

- Logistic Regression is the simplest model and performs well in terms of accuracy (93.87%), but it struggles with the imbalanced dataset, as reflected in its lower F1 score (0.6316).

 ## Metrics Table
| Model                 | Accuracy | F1 Score | Precision | Recall |
|-----------------------|----------|----------|------------|--------|
| Logistic Regression  | 93.8%    | 63.1%    | -          | -      |
| LSTM Model           | 95.5%    | 67.2%    | 69.0%      | 65.5%  |
| Fine-Tuned LSTM      | 95.7%    | 66.8%    | 73.6%      | 61.1%  |
| BERT Transformer     | 93.0%    | -        | -          | -      |


## Usage
- Download the notebook and run the cells
- Save the models; either the traditional model or deep learning model
- Run the model with words to gauge the sentiment of the words


## Conclusion
This project demonstrates the process of performing sentiment analysis on Twitter data using both traditional machine learning and deep learning approaches. The models achieved good accuracy, and the BERT-based model showed promising results. Further fine-tuning and experimentation with different models and hyperparameters could potentially improve performance.

For any questions or issues, please refer to the project documentation or reach out to the project contributors.

**Made by:**
 1. Daniel
 2. Ochan
 3. Xavier



# Customer-Sentiment-Analysis
Sentiment analysis on customer reviews to determine the overall sentiment (positive, negative, neutral).

# Sentiment Analysis on Customer Reviews

## *Muhammad Bakhtiar Salik | Data Analyst*

## Project Overview

This project performs sentiment analysis on customer reviews to classify them into four categories: **Positive, Negative, Neutral, and Irrelevant**. The main goal is to determine the overall sentiment of the reviews based on text data.

## Dataset

The dataset consists of customer reviews labeled with sentiment categories. The text data was preprocessed, vectorized, and used to train a machine learning model.

## Methodology

1. **Data Preprocessing**

   - Removed stopwords and performed tokenization using **spaCy**.
   - Applied lemmatization to convert words to their base form.
   - Removed special characters and converted text to lowercase.

2. **Feature Extraction**

   - Used **TF-IDF (Term Frequency-Inverse Document Frequency)** for vectorization.

3. **Model Training & Evaluation**

   - Tested multiple machine learning models.
   - **Logistic Regression** was found to be the most effective among tested models, achieving an accuracy of around **51%**.
   - Performance metrics included **accuracy, confusion matrix, and classification report**.

## Requirements

Install the necessary libraries before running the project:

```bash
pip install pandas numpy scikit-learn spacy
python -m spacy download en_core_web_sm
```

## How to Run

1. Load the dataset.
2. Preprocess the text using the provided functions.
3. Convert text to numerical features using TF-IDF.
4. Train the model on the training data.
5. Evaluate the model on test data.

## Results

- The trained **Logistic Regression model** achieved the highest accuracy (51%).
- Other models, such as **SVM and XGBoost**, were tested but did not outperform Logistic Regression for this dataset.
- The model can classify reviews into sentiment categories, which can help in analyzing customer feedback trends.

## Future Improvements

- Increase dataset size for better generalization.
- Fine-tune hyperparameters further.
- Explore deep learning approaches such as **LSTMs** or **Transformers**.

## Conclusion

The sentiment analysis model provides insights into customer opinions by classifying reviews into sentiment categories. While improvements can be made, the current implementation offers a solid foundation for text-based sentiment classification.

---

Project Completed 🎯 Presto!


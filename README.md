# 🧠 Twitter Entity Sentiment Analysis with BiLSTM (Keras + Colab)

This project implements a deep learning model using a **Bidirectional LSTM** network to classify the **sentiment** (Positive, Negative, Neutral, Irrelevant) of entities in tweets. It was trained and evaluated using the **Twitter Entity Sentiment Analysis** dataset from Kaggle.

> ✅ Built and trained on Google Colab  
> ✅ Includes model file, tokenizer, and label encoder  
> ✅ Ready for integration in real-time sentiment-based applications

---

## 📂 Project Structure

twitter-sentiment-rnn/
├── TwitterSentimentTraining.ipynb
├── sentiment_model.h5 # Trained BiLSTM model
├── tokenizer.pkl # Tokenizer for input preprocessing
├── label_encoder.pkl # LabelEncoder for target classes
└── README.md 

---

## 📦 Dataset

- **Name**: Twitter Entity Sentiment Analysis  
- **Source**: [Kaggle Dataset](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis)
- **Files**: `twitter_training.csv`, `twitter_validation.csv`  
- **Labels**: `Positive`, `Negative`, `Neutral`, `Irrelevant`

---

## 🚀 Features

- Text preprocessing using regex and NLTK stopwords
- Tokenization and sequence padding using TensorFlow
- Label encoding for multiclass classification
- BiLSTM model with dropout and L2 regularization
- Class weight balancing for imbalanced data
- Evaluation with precision, recall, F1-score, and classification report
- Model, tokenizer, and encoder saved for deployment

---

## 🛠️ Technologies Used

- Python 3.x
- Google Colab
- TensorFlow / Keras
- Scikit-learn
- NLTK
- Pandas / NumPy
- KaggleHub

---

## 🧪 Sample Usage (Colab)

```python
# Load model, tokenizer, and encoder
from tensorflow.keras.models import load_model
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = load_model('sentiment_model.h5')

with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Predict function
def predict_sentiment(text):
    # Clean and preprocess (use same cleaning as training)
    text = text.lower()
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=100)
    pred = np.argmax(model.predict(padded), axis=1)
    return label_encoder.inverse_transform(pred)[0]

# Example
predict_sentiment("I love the new iPhone update!")



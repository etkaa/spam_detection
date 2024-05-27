# spam_email_detection.py

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
@st.cache
def load_data():
    data = pd.read_csv('spam.csv', encoding='latin-1')
    data = data[['v1', 'v2']]
    data.columns = ['label', 'text']
    return data

data = load_data()

# Display the data
st.title("Spam Email Detection")
st.write("Dataset Overview")
st.write(data.head())

if st.checkbox("Show Dataset Summary"):
    st.write(data.describe())

if st.checkbox("Show Data Distribution"):
    st.write(data['label'].value_counts())
    st.bar_chart(data['label'].value_counts())

# Encode labels
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

# Vectorize the text data
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train the model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Make predictions
y_pred = model.predict(X_test_vec)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Model Accuracy: {accuracy:.2f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
st.write("Confusion Matrix")
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
st.pyplot(fig)

# Word frequency (additional visualization)
if st.checkbox("Show Word Frequencies"):
    ham_words = ' '.join(data[data['label'] == 0]['text'])
    spam_words = ' '.join(data[data['label'] == 1]['text'])
    ham_wc = pd.Series(ham_words.split()).value_counts().head(20)
    spam_wc = pd.Series(spam_words.split()).value_counts().head(20)

    st.write("Top words in non-spam emails")
    st.bar_chart(ham_wc)

    st.write("Top words in spam emails")
    st.bar_chart(spam_wc)

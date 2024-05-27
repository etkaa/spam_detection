import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Page layout and styling
st.set_page_config(page_title="Spam Email Detection", layout="wide")
st.title("Spam Email Detection")
st.write("## Overview")
st.write(
    """
    This app uses a Naive Bayes classifier to detect spam emails. 
    It showcases the dataset, provides model evaluation metrics, 
    and allows you to input custom email text for spam detection.
    """
)

# Load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv('spam.csv')
    return data

data = load_data()

# Display the dataset
st.write("### Dataset Preview")
st.dataframe(data.head(), height=150)

# Show dataset description
with st.expander("See dataset statistics"):
    st.write(data.describe())

# Display class distribution
st.write("### Class Distribution")
class_dist = data['spam'].value_counts()
st.bar_chart(class_dist)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    data['text'], data['spam'], test_size=0.2, random_state=42)

# Vectorize the text data
vectorizer = CountVectorizer(stop_words='english', max_features=1000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train the model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Make predictions
y_pred = model.predict(X_test_vec)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
st.write(f"### Model Accuracy: {accuracy:.2f}")

# Confusion matrix
st.write("### Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
st.pyplot(fig)

# Word frequency (additional visualization)
ham_words = ' '.join(data[data['spam'] == 0]['text'])
spam_words = ' '.join(data[data['spam'] == 1]['text'])
ham_wc = pd.Series(ham_words.split()).value_counts().head(20)
spam_wc = pd.Series(spam_words.split()).value_counts().head(20)

st.write("### Top Words in Non-Spam Emails")
st.bar_chart(ham_wc)

st.write("### Top Words in Spam Emails")
st.bar_chart(spam_wc)

# Text input for user to input an email
st.write("### Predict if an Email is Spam or Not")
user_input = st.text_area("Enter the email text here:")
if st.button("Submit"):
    if user_input:
        user_input_vec = vectorizer.transform([user_input])
        prediction = model.predict(user_input_vec)
        prediction_label = "Spam" if prediction[0] == 1 else "Not Spam"
        st.write(f"Prediction: **{prediction_label}**")
    else:
        st.warning("Please enter some text to make a prediction.")

# Hide Streamlit footer and menu
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

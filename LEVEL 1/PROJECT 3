import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = "C:\\Users\\S.Bharathi\\Downloads\\user_reviews.csv\\user_reviews.csv"
data = pd.read_csv(file_path)

# Preprocess the data
def clean_text(text):
    if isinstance(text, str):  # Ensure the text is a string
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = text.lower()
    return text

data['cleaned_text'] = data['Translated_Review'].apply(clean_text).fillna('')

# Encode Sentiments
sentiments = {'Positive': 0, 'Negative': 1, 'Neutral': 2}
data['sentiment_encoded'] = data['Sentiment'].map(sentiments)

# Drop rows with NaN in the sentiment column
data = data.dropna(subset=['sentiment_encoded'])

# Get sentiment counts
sentiment_counts = data['Sentiment'].value_counts()

# Plot sentiment distribution
labels = sentiment_counts.index
counts = sentiment_counts.values

# Define colors for each sentiment
colors = {'Positive': 'green', 'Negative': 'red', 'Neutral': 'blue'}
color_list = [colors[label] for label in labels]

plt.bar(labels, counts, color=color_list)
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.title('Sentiment Analysis of user reviews')
plt.show()

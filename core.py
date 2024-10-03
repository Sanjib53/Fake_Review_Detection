import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import nltk

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')

# Load the CSV file
file_path = 'reviews.csv'  # replace with your file path
df = pd.read_csv(file_path)

# Display the first few rows of the dataframe
print("First few rows of the dataset:")
print(df.head())

# Remove special characters and convert to lowercase
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    return text

# Apply the cleaning function to the reviews column
df['cleaned_review'] = df['review'].apply(clean_text)

# Tokenize the cleaned review
df['tokenized_review'] = df['cleaned_review'].apply(word_tokenize)

# Optionally remove stopwords
stop_words = set(stopwords.words('english'))
df['filtered_review'] = df['tokenized_review'].apply(lambda x: [word for word in x if word not in stop_words])

# Example: Add a feature for review length
df['review_length'] = df['cleaned_review'].apply(len)

# Example: Add a feature for the number of exclamation marks
df['exclamation_marks'] = df['review'].apply(lambda x: x.count('!'))

print("\nData with additional features:")
print(df[['review', 'cleaned_review', 'review_length', 'exclamation_marks']].head())

# Vectorize the text data
vectorizer = CountVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['cleaned_review']).toarray()

# Assuming you have a target column 'label' where 1 = fake, 0 = real
y = df['label']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model with the 'saga' solver and increased iterations
model = LogisticRegression(solver='saga', max_iter=5000, class_weight='balanced')
model.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate the model
print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
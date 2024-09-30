import os
import tarfile
import urllib.request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# load dataset
url = "https://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz"
dataset_path = "rt-polaritydata.tar.gz"
if not os.path.exists(dataset_path):
    urllib.request.urlretrieve(url, dataset_path)

with tarfile.open(dataset_path, "r:gz") as tar:
    tar.extractall()

positive_path = "rt-polaritydata/rt-polarity.pos"
negative_path = "rt-polaritydata/rt-polarity.neg"

with open(positive_path, 'r', encoding='latin-1') as f:
    positive_sentences = f.readlines()

with open(negative_path, 'r', encoding='latin-1') as f:
    negative_sentences = f.readlines()

train_pos, val_pos, test_pos = positive_sentences[:4000], positive_sentences[4000:4500], positive_sentences[4500:]
train_neg, val_neg, test_neg = negative_sentences[:4000], negative_sentences[4000:4500], negative_sentences[4500:]

train_data = train_pos + train_neg
train_labels = [1] * 4000 + [0] * 4000

val_data = val_pos + val_neg
val_labels = [1] * 500 + [0] * 500

test_data = test_pos + test_neg
test_labels = [1] * 831 + [0] * 831

# create features
vectorizer = CountVectorizer(binary=True)
X_train = vectorizer.fit_transform(train_data)
X_test = vectorizer.transform(test_data)

"""# **1. Logistic Regression Classifier**"""

from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, train_labels)

val_predictions = lr_model.predict(vectorizer.transform(val_data))

lr_predictions = lr_model.predict(X_test)

# metrics
tn, fp, fn, tp = confusion_matrix(test_labels, lr_predictions).ravel()
precision = precision_score(test_labels, lr_predictions)
recall = recall_score(test_labels, lr_predictions)
f1 = f1_score(test_labels, lr_predictions)

print(f"Logistic Regression Classifier Metrics:")
print(f"True Positive (TP): {tp}")
print(f"True Negative (TN): {tn}")
print(f"False Positive (FP): {fp}")
print(f"False Negative (FN): {fn}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")

# confusion matrix
conf_matrix = confusion_matrix(test_labels, lr_predictions)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Greens')
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# classification report
print("Classification Report:")
print(classification_report(test_labels, lr_predictions))

"""# **2. Naive Bayes Classifier**"""

from sklearn.naive_bayes import MultinomialNB

nb_model = MultinomialNB()
nb_model.fit(X_train, train_labels)

val_predictions = nb_model.predict(vectorizer.transform(val_data))

nb_predictions = nb_model.predict(X_test)

# metrics
tn, fp, fn, tp = confusion_matrix(test_labels, nb_predictions).ravel()
precision = precision_score(test_labels, nb_predictions)
recall = recall_score(test_labels, nb_predictions)
f1 = f1_score(test_labels, nb_predictions)

print(f"Naive Bayes Classifier Metrics:")
print(f"True Positive (TP): {tp}")
print(f"True Negative (TN): {tn}")
print(f"False Positive (FP): {fp}")
print(f"False Negative (FN): {fn}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")

# confusion matrix
conf_matrix = confusion_matrix(test_labels, nb_predictions)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Greens')
plt.title("Confusion Matrix - Naive Bayes")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# classification report
print("Classification Report:")
print(classification_report(test_labels, nb_predictions))

"""# **3. Recurrent Neural Network (RNN)**"""

import tensorflow as tf
import numpy as np

max_words = 1000
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(train_data)
X_train_seq = tokenizer.texts_to_sequences(train_data)
X_val_seq = tokenizer.texts_to_sequences(val_data)
X_test_seq = tokenizer.texts_to_sequences(test_data)

X_train_padded = tf.keras.preprocessing.sequence.pad_sequences(X_train_seq, maxlen=100)
X_val_padded = tf.keras.preprocessing.sequence.pad_sequences(X_val_seq, maxlen=100)
X_test_padded = tf.keras.preprocessing.sequence.pad_sequences(X_test_seq, maxlen=100)

rnn_model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=max_words, output_dim=64, input_length=100),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

rnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
rnn_model.fit(X_train_padded, np.array(train_labels), epochs=5, batch_size=64)

val_predictions = (rnn_model.predict(X_val_padded) > 0.5).astype("int32").flatten()

rnn_predictions = (rnn_model.predict(X_test_padded) > 0.5).astype("int32").flatten()

# metrics
tn, fp, fn, tp = confusion_matrix(test_labels, rnn_predictions).ravel()
precision = precision_score(test_labels, rnn_predictions)
recall = recall_score(test_labels, rnn_predictions)
f1 = f1_score(test_labels, rnn_predictions)

print(f"RNN Classifier Metrics:")
print(f"True Positive (TP): {tp}")
print(f"True Negative (TN): {tn}")
print(f"False Positive (FP): {fp}")
print(f"False Negative (FN): {fn}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")

# confusion matrix
conf_matrix = confusion_matrix(test_labels, rnn_predictions)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Greens')
plt.title("Confusion Matrix - RNN")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# classification report
print("Classification Report:")
print(classification_report(test_labels, rnn_predictions))
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
import pickle
import string

# Loading Spacy small model as nlp
nlp = spacy.load("en_core_web_lg")

from spacy.lang.en.stop_words import STOP_WORDS

stopwords = list(STOP_WORDS)
print(len(stopwords))

# Adding yelp dataset
data_yelp = pd.read_csv('./data/datasets/yelp_labelled.txt',
                        sep='\t', header=None)
data_yelp.head()

columnName = ['Review', 'Sentiment']
data_yelp.columns = columnName
data_yelp.head()

# Adding Amazon dataset and adding its column name
data_amz = pd.read_csv("./data/datasets/amazon_cells_labelled.txt",
                       sep='\t', header=None)
data_amz.columns = columnName
data_amz.head()

# Adding IMdB dataset and adding its column name
data_imdb = pd.read_csv("./data/datasets/imdb_labelled.txt",
                        sep='\t', header= None)
data_imdb.columns = columnName
data_imdb.head()


#Append all datasets
data = data_yelp.append([data_amz, data_imdb], ignore_index=True)
punct = string.punctuation


# Spillting the train and test data
X = data['Review']
y = data['Sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Creating the model and pipeline
tfidf = TfidfVectorizer(tokenizer=dataCleaning)
svm = LinearSVC()
steps = [('tfidf', tfidf), ('svm', svm)]
pipe = Pipeline(steps)

print("Training model\n")
# Training the model
pipe.fit(X_train, y_train)

# Testing on the test dataset
y_pred = pipe.predict(X_test)


# Printing the classification report and the confusion matrix
print(classification_report(y_test, y_pred))
print("\n\n")
print(confusion_matrix(y_test, y_pred))


filename = './data/sentiment.dat'
pickle.dump(pipe, open(filename, 'wb'))
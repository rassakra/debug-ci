import base64
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
import json
import string
import spacy
import pandas as pd
import unicodedata
from flask import Flask, request
from flask import Response
from spacy.language import Language
from urllib.parse import unquote
import pickle
from spacytextblob.spacytextblob import SpacyTextBlob
from spacy.lang.en.stop_words import STOP_WORDS

PORT = 8190
app = Flask(__name__)

spacy.prefer_gpu()
stopwords = list(STOP_WORDS)
punct = string.punctuation


nlp = spacy.load('en_core_web_lg')

def trainModel():
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

    # Adding Twitter dataset and adding its column name
    data_twitter = pd.read_csv("./data/datasets/twitter_labelled.csv", encoding='latin', names=['Sentiment', 'id', 'date', 'query','user', 'Review'])
    data_twitter['Sentiment']=data_twitter['Sentiment'].replace(4,1)
    data_twitter.drop(['date','query','user'], axis=1, inplace=True)
    data_twitter.drop('id', axis=1, inplace=True)
    data_twitter['Review'] = data_twitter['Review'].astype('str')
    data_twitter.head()


    #Append all datasets
    data = data_yelp.append([data_amz, data_imdb, data_twitter], ignore_index=True)


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
    return pipe


def dataCleaning(sentence):
    doc = nlp(sentence)
    tokens = []
    for token in doc:
        if token.lemma_ != '-PRON-':
            temp = token.lemma_.lower().strip()
        else:
            temp = token.lower_
        tokens.append(temp)
    clean_tokens = []
    for token in tokens:
        if token not in punct and token not in stopwords:
            clean_tokens.append(token)
    return clean_tokens

@Language.component("set_custom_boundaries")
def set_custom_boundaries(doc):
    for token in doc[:-1]:
        if token.text == "\n":
            doc[token.i].is_sent_start = True
    return doc


nlp.add_pipe("set_custom_boundaries", before="parser")

rulerConfig = {
    "validate": True,
    "overwrite_ents": True,
}

nlp.add_pipe("spacytextblob")

# TODO Swap between load/train
# loaded_model = trainModel()
loaded_model = pickle.load(open('./data/sentiment.dat', 'rb'))

@app.route('/')
def search():
    key = unquote(request.args.get('q'))
    res = base64.b64decode(key).decode('utf8', 'replace')
    # res= key
    res = unicodedata.normalize("NFKD", res)
    print("Repr: " + repr(res))

    doc = nlp(res)

    prediction = loaded_model.predict([res])
    print(prediction)

    return Response(json.dumps({
        "entities": [
            {"text": ent.text, "start": ent.start_char, "end": ent.end_char, "label": ent.label_, "id": ent.ent_id_,
             "lemma": ent.lemma_}
            for ent in doc.ents
        ],
        "subjects": [
            {"chunk": chunk.text}
            for chunk in doc.noun_chunks
        ],
        "polarity": doc._.polarity,
        "subjectivity": doc._.subjectivity,
        "assesments": doc._.assessments
    }), mimetype='application/json')


if __name__ == "__main__":
    app.run(host='0.0.0.0')

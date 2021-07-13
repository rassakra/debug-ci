import base64
from spacytextblob.spacytextblob import SpacyTextBlob
import json
import string
import spacy
import unicodedata
from flask import Flask, request
from flask import Response
from spacy.language import Language
from urllib.parse import unquote

PORT = 8190
app = Flask(__name__)

spacy.prefer_gpu()
punct = string.punctuation


nlp = spacy.load('en_core_web_lg')

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

@app.route('/')
def search():
    key = unquote(request.args.get('q'))
    res = base64.b64decode(key).decode('utf8', 'replace')
    # res= key
    res = unicodedata.normalize("NFKD", res)
    print("Repr: " + repr(res))

    doc = nlp(res)

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

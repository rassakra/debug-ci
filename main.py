#!/usr/bin/env python3
# coding=utf-8

import base64
import json
import spacy
import textacy
import unicodedata
from flask import Flask, request
from flask import Response
from urllib.parse import unquote
from spacy.language import Language

PORT = 8180
app = Flask(__name__)

spacy.prefer_gpu()
nlp = spacy.load("en_core_web_lg")

config = {
    "punct_chars": ['.', '?', '!', ',\n', '*', '- ', ':\n', '۔', '܀',
                    '܁', '܂', '߹',
                    '।', '॥', '၊', '။', '።', '፧', '፨', '᙮', '᜵', '᜶', '᠃', '᠉', '᥄',
                    '᥅', '᪨', '᪩', '᪪', '᪫', '᭚', '᭛', '᭞', '᭟', '᰻', '᰼', '᱾', '᱿',
                    '‼', '‽', '⁇', '⁈', '⁉', '⸮', '⸼', '꓿', '꘎', '꘏', '꛳', '꛷', '꡶',
                    '꡷', '꣎', '꣏', '꤯', '꧈', '꧉', '꩝', '꩞', '꩟', '꫰', '꫱', '꯫', '﹒',
                    '﹖', '﹗', '！', '．', '？', '𐩖', '𐩗', '𑁇', '𑁈', '𑂾', '𑂿', '𑃀',
                    '𑃁', '𑅁', '𑅂', '𑅃', '𑇅', '𑇆', '𑇍', '𑇞', '𑇟', '𑈸', '𑈹', '𑈻', '𑈼',
                    '𑊩', '𑑋', '𑑌', '𑗂', '𑗃', '𑗉', '𑗊', '𑗋', '𑗌', '𑗍', '𑗎', '𑗏', '𑗐',
                    '𑗑', '𑗒', '𑗓', '𑗔', '𑗕', '𑗖', '𑗗', '𑙁', '𑙂', '𑜼', '𑜽', '𑜾', '𑩂',
                    '𑩃', '𑪛', '𑪜', '𑱁', '𑱂', '𖩮', '𖩯', '𖫵', '𖬷', '𖬸', '𖭄', '𛲟', '𝪈',
                    '｡', '。']}

nlp.create_pipe('sentencizer', config=config)

sentencizer = nlp.add_pipe('sentencizer', before="parser")

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

entity_ruler = nlp.add_pipe("entity_ruler", config=rulerConfig)
entity_ruler.from_disk("./data/va_train.jsonl")

print(nlp.pipe_names)


@app.route('/')
def search():
    key = unquote(request.args.get('q'))
    res = base64.b64decode(key).decode('utf8', 'replace')
    res = unicodedata.normalize("NFKD", res)

    print(repr(res))

    doc = textacy.make_spacy_doc(res, lang=nlp)

######## DONT CHANGE
    return Response(json.dumps({
        "words": [
            {"text": token.text, "data_type": token.pos_, "dep": token.dep_, "parent": token.head.text,
             "lemma": token.lemma_, "tag": token.tag_}
            for token in doc
        ],
        "entities": [
            {"text": ent.text, "start": ent.start_char, "end": ent.end_char, "label": ent.label_, "id": ent.ent_id_,
             "lemma": ent.lemma_}
            for ent in doc.ents
        ],
        "sentances": [
            {"text": s.text, "start": s.start_char, "end": s.end_char, "label": s.label}
            for s in doc.sents
        ]
    }), mimetype='application/json')


if __name__ == "__main__":
    app.run(host='0.0.0.0')

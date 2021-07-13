Virtual Assistant NLP Server


In order to integrate with the VA Service, the following payload must be returned

```json
{
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
    }
```



### Installation

`pip install -r requirements.txt`
`python -m spacy download en_core_web_lg`

If you have errors installing with `[Erro No2]` its most likely due to path length of the install. to fix run the following:
1. Type `regedit` in the Windows start menu to launch regedit.
2. Go to the `Computer\HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem` key.
3. Edit the value of the `LongPathsEnabled` property of that key and set it to 1.
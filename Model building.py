import string
import pandas as pd
from sklearn import feature_extraction
from sklearn import linear_model
from sklearn import pipeline
from sklearn import metrics
from sklearn.model_selection import train_test_split
import re

df = pd.read_csv("Language Detection.csv")


df_eng = df[df['Language'] == 'English']['Text']


df_ger = df[df['Language'] == 'German']['Text']


df_spa = df[df['Language'] == 'Spanish']['Text']


df_fre = df[df['Language'] == 'French']['Text']


df_dutch = df[df['Language'] == 'Dutch']['Text']


df_danish = df[df['Language'] == 'Danish']['Text']


df_it = df[df['Language'] == 'Italian']['Text']


for char in string.punctuation:
    print(char, end=" ")
translate_table = dict((ord(char), None) for char in string.punctuation)

data_eng = []
lang_eng = []
for line in df_eng:
    if len(line) != 0:
        line = line.lower()
        line = re.sub(r"\d+", "", line)
        line = line.translate(translate_table)
        data_eng.append(line)
        lang_eng.append("English")

print(data_eng)
print(lang_eng)

data_ger = []
lang_ger = []
for line in df_ger:
    if len(line) != 0:
        line = line.lower()
        line = re.sub(r"\d+", "", line)
        line = line.translate(translate_table)
        data_ger.append(line)
        lang_ger.append("German")

print(data_ger)
print(lang_ger)

data_dutch = []
lang_dutch = []
for line in df_dutch:
    if len(line) != 0:
        line = line.lower()
        line = re.sub(r"\d+", "", line)
        line = line.translate(translate_table)
        data_dutch.append(line)
        lang_dutch.append("Dutch")

print(data_dutch)
print(lang_dutch)

data_spa = []
lang_spa = []
for line in df_spa:
    if len(line) != 0:
        line = line.lower()
        line = re.sub(r"\d+", "", line)
        line = line.translate(translate_table)
        data_spa.append(line)
        lang_spa.append("Spanish")

print(data_spa)
print(lang_spa)

data_fre = []
lang_fre = []
for line in df_fre:
    if len(line) != 0:
        line = line.lower()
        line = re.sub(r"\d+", "", line)
        line = line.translate(translate_table)
        data_fre.append(line)
        lang_fre.append("French")

print(data_fre)
print(lang_fre)

data_danish = []
lang_danish = []
for line in df_danish:
    if len(line) != 0:
        line = line.lower()
        line = re.sub(r"\d+", "", line)
        line = line.translate(translate_table)
        data_danish.append(line)
        lang_danish.append("Danish")

print(data_danish)
print(lang_danish)

data_it = []
lang_it = []
for line in df_it:
    if len(line) != 0:
        line = line.lower()
        line = re.sub(r"\d+", "", line)
        line = line.translate(translate_table)
        data_it.append(line)
        lang_it.append("Italian")

print(data_it)
print(lang_it)

final_data = pd.DataFrame({"Text": data_eng + data_ger + data_fre + data_spa + data_dutch + data_danish + data_it,
                           "Language": lang_eng + lang_ger + lang_fre + lang_spa + lang_dutch + lang_danish + lang_it})

print(final_data.shape)

final_data.head()

x, y = final_data.iloc[:, 0], final_data.iloc[:, 1]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

vectorizer = feature_extraction.text.TfidfVectorizer(ngram_range=(1, 3), analyzer='char')

pipe_lr_r13 = pipeline.Pipeline([
    ('vectorizer', vectorizer),
    ('clf', linear_model.LogisticRegression())
])

pipe_lr_r13.fit(X_train, y_train)

y_predicted = pipe_lr_r13.predict(X_test)

acc = (metrics.accuracy_score(y_test, y_predicted)) * 100
print(acc, '%')

matrix = metrics.confusion_matrix(y_test, y_predicted)
print('Confusion matrix: \n', matrix)

import pickle

lrFile = open('LRModel.pckl', 'wb')
pickle.dump(pipe_lr_r13, lrFile)
lrFile.close()


def lang_detect(text):
    import numpy as np
    import string
    import re
    import pickle
    translate_table = dict((ord(char), None) for char in string.punctuation)

    global lrLangDetectionModel
    lrLangDetectionFile = open('LRModel.pckl', 'rb')
    lrLangDetectionModel = pickle.load(lrLangDetectionFile)
    lrLangDetectionFile.close()

    text = " ".join(text.split())
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(translate_table)
    pred = lrLangDetectionModel.predict([text])
    prob = lrLangDetectionModel.predict_proba([text])
    return pred[0]


lang_detect("Wir haben die Keks gebracht.")

lang_detect("Quiero aprender español")

lang_detect(
    "Questo premio fu accompagnato da un finanziamento di 10.000 euro e da un invito a presenziare al PAE Cyberarts Festival di quell'anno.")

lang_detect("Hello how are you?")

lang_detect("Hej! Jeg hedder Morten. Hvad hedder du?")

lang_detect("Apprendre à lire")

lang_detect("Waar komt u vandaan?")

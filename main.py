import string
import re
import pickle
import string


def lang_detect(text):
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


def main():
    text = "PUT YOUR TEXT HERE"
    detect = lang_detect(text)
    print("Language Detected: ", detect)


if __name__ == "__main__":
    main()

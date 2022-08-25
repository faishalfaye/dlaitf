import json
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences



def parse_data(file):
    for l in open(file, 'r'):
        yield json.loads(l) #kalo pake return nanti cuma return file row pertama doang

data = list(parse_data('C:\\Users\\Win10\\PycharmProjects\\Sarcasm_Headlines_Dataset.json'))
# print(data)
headline  = []
for item in data:
    headline.append(item["headline"])

tokenizer = Tokenizer(oov_token= "<OOV>")
tokenizer.fit_on_texts(headline)

word_index = tokenizer.word_index
print(word_index)

sequences = tokenizer.texts_to_sequences(headline)
padded = pad_sequences(sequences, padding="post", truncating="post")
print(headline[556])
print(padded[556])
print(padded.shape)





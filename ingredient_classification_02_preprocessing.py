import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import pickle
import re

pd.set_option('display.unicode.east_asian_width', True)

df = pd.read_csv('./crawling/concat_recipes.csv')

X = df['Ingredient']
Y = df['Category']
print(X)

encoder = LabelEncoder()
labeled_Y = encoder.fit_transform(Y)
label = encoder.classes_

print(labeled_Y[1000])
print(label)
with open('./models/encoder.pickle', 'wb') as f:
    pickle.dump(encoder, f)
onehot_Y = to_categorical(labeled_Y)
print(onehot_Y[1000])

okt = Okt()
for i in range(len(X)):
    X[i] = re.compile('[^가-힣]').sub(' ', X[i])
    X[i] = okt.morphs(X[i])
    # X[i] = okt.nouns(X[i])
print(X)

# 자연어처리가 안되어있는 것들이 있어서 re.compile추가

stopwords = pd.read_csv('./crawling/stopword_final.csv', index_col=0)
print(stopwords)

for j in range(len(X)):
    words = []
    for k in range(len(X[j])):
        if X[j][k] not in list(stopwords['stopword']):
            # X[j][k] = re.compile('[^가-힣a-zA-Z]').sub(' ', X[j][k])
            words.append(X[j][k])
    X[j] = ' '.join(words)
print(X)
print(type(X))
print(X[0])



token = Tokenizer()
token.fit_on_texts(X)
tokened_X = token.texts_to_sequences(X)
print(tokened_X[:5])
print(X[:5])

with open('./models/recipes_token.pickle', 'wb') as f:
    pickle.dump(token, f)

# token을 통해 형태소에 숫자를 부여(라벨링)

wordsize = len(token.word_index) + 1
print(wordsize)
print(token.index_word)

max = 0
for i in range(len(tokened_X)):
    if max < len(tokened_X[i]):
        max = len(tokened_X[i])
print(max)

X_pad = pad_sequences(tokened_X, max)

# padding으로 0 채워서 길이 맞춰주기(for input_dim)

X_train, X_test, Y_train, Y_test = train_test_split(
    X_pad, onehot_Y, test_size=0.1)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

xy = X_train, X_test, Y_train, Y_test
# np.save('./crawling/recipes_max_{}_wordsize_{}'.format(max, wordsize),xy)


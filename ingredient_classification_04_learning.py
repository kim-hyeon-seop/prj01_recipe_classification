

import pandas as pd
import numpy as np
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import pickle
import re

pd.set_option('display.unicode.east_asian_width', True)



# 데이터로드
df = pd.read_csv('./crawling/concat_recipes.csv')

print(df.head())
print(df.info())
print(df.Category.value_counts())
print(df.loc[df['Category']=='Japanese'])

X = df['Ingredient']
Y = df['Category']



# target labeling
with open('./models/encoder.pickle', 'rb') as f:
    encoder = pickle.load(f)


labeled_Y = encoder.transform(Y)
label = encoder.classes_
print(labeled_Y[:5])
print(label)



onehot_Y = to_categorical(labeled_Y)
print(onehot_Y)



# 형태소 분리, 한 글자/불용어 제거
okt = Okt()

for i in range(len(X)):
    X[i] = re.compile('[^가-힣]').sub(' ', X[i])
    X[i] = okt.morphs(X[i], stem=True)
print(X)


stopwords = pd.read_csv('./crawling/stopword_final.csv',
                        index_col=0)

for j in range(len(X)):
    words = []
    for i in range(len(X[j])):
            if X[j][i] not in list(stopwords['stopword']):
                words.append(X[j][i])
    X[j] = ' '.join(words)
print(X)



# tokenizing
with open('./models/recipes_token.pickle', 'rb') as f:
    token = pickle.load(f)

tokened_X = token.texts_to_sequences(X)
print(tokened_X[:5])

max = 0
for i in range(len(tokened_X)):
    if 70 < len(tokened_X[i]):
        tokened_X[i] = tokened_X[i][:70]

print(max)





# padding
X_pad = pad_sequences(tokened_X, 70)
print(X_pad[:10])

# load model
from tensorflow.keras.models import load_model

model = load_model('./models/recipe_classification_model_0.781862735748291.h5')
preds = model.predict(X_pad)
predicts = []
for pred in preds:
    predicts.append(label[np.argmax(pred)])
df['predict'] = predicts
pd.set_option('display.unicode.east_asian_width', True)
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 100)
print(df.head(30))

df['OX'] = 0
for i in range(len(df)):
    if df.loc[i,'Category'] == df.loc[i, 'predict']:
        df.loc[i, 'OX'] = 'O'
    else:
        df.loc[i, 'OX'] = 'X'
print(df[1000:1100])
print(df['OX'].value_counts())
df.to_csv('./crawling/predict.csv')

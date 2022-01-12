# 새로 크로울링한 헤드라인에 대해서 모델한테 예측을 시켜보자.

# 지금할 것. preprocessing을 해야해.

# 크로울링은 해왔지. X, Y로 나눠주는거 하고, 라벨인코딩,
# 오픈으로 엔코더 열고 transform하자. exam27

# 토큰은 피클담근거 진행하지 말고, 워드사이즈도 필요없음. 이미
# 토크나이징할때 정해진거거든.
# 모델이 학습할 때 최대길이로 학습했으니 맞춰줘야돼.
# 새로 크로울링한 애도 23으로 맞춰줘야돼. 길이가 23 넘어가면 신경쓰자
# 단어 사전에 없는 녀석들은 번역 못해서 없어질 것

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
# 파일로 인코더를 받아서 여는 것

labeled_Y = encoder.transform(Y)
label = encoder.classes_
print(labeled_Y[:5])
print(label)

# 앞에서 fit_transform 했으니 fit 빼고 transform만 해야해.

onehot_Y = to_categorical(labeled_Y)
print(onehot_Y)



# 형태소 분리, 한 글자/불용어 제거
okt = Okt()

for i in range(len(X)):
    X[i] = re.compile('[^가-힣]').sub(' ', X[i])
    X[i] = okt.morphs(X[i], stem=True)
print(X)
# stem=True는 어간만 잘라줌

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
# 토큰 불러오자

tokened_X = token.texts_to_sequences(X)
print(tokened_X[:5])

max = 0
for i in range(len(tokened_X)):
    if 70 < len(tokened_X[i]):
        tokened_X[i] = tokened_X[i][:70]
    # 길면 23글자 이하로 잘라버린다.
print(max)

# max값은 정해진 것(71) 그대로 써야해
# 워드 목록에 없는 단어들은 texts_to_sequence할 때 버려져. 있는 애들만 토크나이징




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

# (label[np.argmax(pred[-1])])이게 모델이 예측한 값. for문을 통해 비교교
# label은 아까 encoder.classes_에서 어떤 값을 숫자 몇으로 바꿨는지에 대한
# 정보가 있어.

# 얼마나 맞췄는지 봅시다. df에 컬럼 하나 더 만들기
df['OX'] = 0
for i in range(len(df)):
    if df.loc[i,'Category'] == df.loc[i, 'predict']:
        df.loc[i, 'OX'] = 'O'
    else:
        df.loc[i, 'OX'] = 'X'
print(df[1000:1100])
print(df['OX'].value_counts())
df.to_csv('./crawling/predict.csv')

# 하면 정답률 87퍼센트
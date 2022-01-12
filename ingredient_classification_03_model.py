import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *

X_train, X_test, Y_train, Y_test = np.load(
    './crawling/recipes_max_70_wordsize_2062.npy',
    allow_pickle = True)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)


model = Sequential()
model.add(Embedding(2062, 300, input_length=70))
model.add(Conv1D(32, kernel_size = 5, padding='same',
                 activation='relu'))
model.add(MaxPool1D(pool_size=1))
model.add(LSTM(128,activation = 'tanh', return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(64,activation = 'tanh', return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(64,activation = 'tanh'))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dense(4, activation='softmax'))
print(model.summary())



model.compile(loss='categorical_crossentropy',
              optimizer = 'adam', metrics=['accuracy'])
fit_hist = model.fit(X_train, Y_train, batch_size=100,
                     epochs=20, validation_data=(X_test,Y_test))

model.save('./models/recipe_classification_model_{}.h5'.format(
    fit_hist.history['val_accuracy'][-1]))

plt.plot(fit_hist.history['accuracy'], label='accuracy')
plt.plot(fit_hist.history['val_accuracy'], label='val accuracy')
plt.legend()
plt.show()

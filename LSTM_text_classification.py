###################### load packages ####################
from keras.datasets import imdb
from keras import preprocessing
from keras.models import Sequential
from keras.layers import Dense, Embedding, Dropout, LSTM
from keras.utils.np_utils import to_categorical


###################### load data ####################
######### 只考虑最常见的1000个词 ########
num_words = 1000

######### 导入数据 #########
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)

print(x_train.shape)
print(x_train[0][:5])

print(y_train.shape)
print(y_train[0])


###################### preprocess data ####################
######## 句子长度最长设置为20 ########
max_len = 20

######## 对文本进行填充，将文本转成相同长度 ########
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=max_len)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=max_len)

print(x_train.shape)
print(x_train[0])

######## 对label做one-hot处理 ########
num_class = 2
y_train = to_categorical(y_train, num_class)
y_test = to_categorical(y_test, num_class)

print(y_train.shape)
print(y_train[0])


###################### build network ####################
######## word dim 词向量维度 ########
word_dim = 8

######## network structure ########
model = Sequential()

#### Embedding层 ####
model.add(Embedding(input_dim=1000, output_dim=word_dim, input_length=max_len))

#### 两层LSTM，第一层，设置return_sequences参数为True ####
model.add(LSTM(256, return_sequences=True))

#### dropout ####
model.add(Dropout(0.5))

#### 两层LSTM，第二层，设置return_sequences参数为False ####
model.add(LSTM(256, return_sequences=False))

#### dropout ####
model.add(Dropout(0.5))

#### 输出层 ####
model.add(Dense(num_class, activation='softmax'))

print(model.summary())

######## optimization and train ########
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, batch_size=512, epochs=20, verbose=1, validation_data=(x_test, y_test))

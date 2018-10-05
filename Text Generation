# 数据清洗
import pandas as pd
df = pd.read_csv(u'test.csv')
story = ''.join(str(i) for i in df['text']) # len(story)=573672

# 建立词表及索引
cate = set(story)
cate_num = len(cate) # 3453类字符
char2index = dict((c, i) for i, c in enumerate(cate)) # {'字符': '索引'}
index2char = dict((i, c) for i, c in enumerate(cate)) # {'索引': '字符'}

# 定义20个字符串为‘样本’，其后一个字符串为该样本对应的‘标签’
seqlen = 20
step = 1
input_chars = []
label_chars = []
for i in range(0, len(story) - seqlen, step):
    input_chars.append(story[i:i+seqlen])   # len(input_chars)=len(story)-seqlen=573672-20=573652
    label_chars.append(story[i+seqlen])     # len(label_chars)=len(input_chars)=573652
    
# 为样本序列赋向量值（独热表示）
import numpy as np
X = np.zeros((len(input_chars), seqlen, cate_num),dtype=np.bool)
y = np.zeros((len(label_chars), cate_num),dtype=np.bool)
for i, input_char in enumerate(input_chars):
    y[i, char2index[label_chars[i]]] = 1
    for j, ch in enumerate(input_char):
        X[i, j, char2index[ch]] = 1

HIDDEN_SIZE = 128
BATCH_SIZE = 500
NUM_ITERATIONS = 50
NUM_EPOCHS_PER_ITERATION = 1
NUM_PREDS_PER_EPOCH = 50

from keras.layers.recurrent import SimpleRNN
from keras.models import Sequential
from keras.layers import Dense, Activation

model = Sequential()
model.add(SimpleRNN(HIDDEN_SIZE, return_sequences=False,
                    input_shape=(seqlen, cate_num),
                    unroll=True))
model.add(Dense(cate_num))
model.add(Activation("softmax"))
model.compile(loss="categorical_crossentropy", optimizer="rmsprop")


for iteration in range(NUM_ITERATIONS):
    print("=" * 50)
    print("Iteration #: %d" % (iteration))
    model.fit(X, y, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS_PER_ITERATION)
    test_idx = np.random.randint(len(input_chars))
    test_chars = input_chars[test_idx]
    print("Generating from seed: %s" % (test_chars))
    print(test_chars, end="")
    for i in range(NUM_PREDS_PER_EPOCH): # 循环100次，每次循环去掉第一个字符，加上新的预测字符
        Xtest = np.zeros((1, seqlen, cate_num))
        for i, ch in enumerate(test_chars):
            Xtest[0, i, char2index[ch]] = 1
        pred = model.predict(Xtest, verbose=0)[0]
        ypred = index2char[np.argmax(pred)]
        print(ypred, end="")
        test_chars = test_chars[1:] + ypred
    print()

循环神经网络详解
==============

# 一. 概述

> CNN等传统神经网络的局限在于：将固定大小的向量作为输入（比如一张图片），然后输出一个固定大小的向量（比如不同分类的概率）。不仅如此，CNN还按照固定的计算步骤（比如模型中层的数量）来实现这样的输入输出。这样的神经网络没有持久性：假设你希望对电影中每一帧的事件类型进行分类，传统的神经网络就没有办法使用电影中先前的事件推断后续的事件。
> RNN 解决了这个问题。RNN 是包含循环的网络，允许信息的持久化。在自然语言处理(NLP)领域，RNN已经可以做语音识别、机器翻译、生成手写字符，以及构建强大的语言模型 (Sutskever et al.)，(Graves)，(Mikolov et al.)（字符级别和单词级别的都有。在机器视觉领域，RNN也非常流行。包括帧级别的视频分类，图像描述，视频描述以及基于图像的Q&A等等。

# 二. 结构

> RNN结构如下图所示：

![image](https://github.com/ShaoQiBNU/RNN/blob/master/images/1.png)

> 神经网络的模块A正在读取某个输入xt，并输出一个值ht，循环可以使得信息从当前步传递到下一步，将这个循环展开，如下所示。链式的特征揭示了 RNN 本质上是与序列和列表相关的，它们是对于这类数据的最自然的神经网络架构。

![image](https://github.com/ShaoQiBNU/RNN/blob/master/images/2.png)

# 三. 长期依赖问题

> RNN 的关键点之一就是他们可以用来连接先前的信息到当前的任务上，例如使用过去的视频段来推测对当前段的理解。如果 RNN 可以做到这个，他们就变得非常有用。但是真的可以么？答案是，还有很多依赖因素。有时候，我们仅仅需要知道先前的信息来执行当前的任务。例如，我们有一个语言模型用来基于先前的词来预测下一个词。如果我们试着预测 “the clouds are in the sky” 最后的词，我们并不需要任何其他的上下文 —— 因此下一个词很显然就应该是 sky。在这样的场景中，相关的信息和预测的词位置之间的间隔是非常小的，RNN 可以学会使用先前的信息。

![image](https://github.com/ShaoQiBNU/RNN/blob/master/images/3.png)

> 但是同样会有一些更加复杂的场景。假设我们试着去预测“I grew up in France... I speak fluent French”最后的词。当前的信息建议下一个词可能是一种语言的名字，但是如果我们需要弄清楚是什么语言，我们是需要先前提到的离当前位置很远的 France 的上下文的。这说明相关信息和当前预测位置之间的间隔就肯定变得相当的大。不幸的是，在这个间隔不断增大时，RNN 会丧失学习到连接如此远的信息的能力。

![image](https://github.com/ShaoQiBNU/RNN/blob/master/images/4.png)

> 在理论上，RNN 绝对可以处理这样的长期依赖问题。人们可以仔细挑选参数来解决这类问题中的最初级形式，但在实践中，RNN 肯定不能够成功学习到这些知识。Bengio, et al. (1994)等人对该问题进行了深入的研究，他们发现一些使训练 RNN 变得非常困难的相当根本的原因。然而，幸运的是，LSTM 并没有这个问题！

# 四. LSTM网络

> Long Short Term 网络—— 一般就叫做 LSTM，是一种 RNN 特殊的类型，可以学习长期依赖信息。LSTM 由Hochreiter & Schmidhuber (1997)提出，并在近期被Alex Graves进行了改良和推广。在很多问题，LSTM 都取得相当巨大的成功，并得到了广泛的使用。LSTM 通过刻意的设计来避免长期依赖问题。记住长期的信息在实践中是 LSTM 的默认行为，而非需要付出很大代价才能获得的能力！

> 所有 RNN 都具有一种重复神经网络模块的链式的形式。在标准的 RNN 中，这个重复的模块只有一个非常简单的结构，例如一个 tanh 层。

![image](https://github.com/ShaoQiBNU/RNN/blob/master/images/5.png)

> LSTM 同样是这样的结构，但是重复的模块拥有一个不同的结构。不同于 单一神经网络层，这里是有四个，以一种非常特殊的方式进行交互。

![image](https://github.com/ShaoQiBNU/RNN/blob/master/images/6.png)

# 五. LSTM网络详解

> 下面对LSTM网络进行详细说明，首先说明一下图中使用的图标，如下：

![image](https://github.com/ShaoQiBNU/RNN/blob/master/images/7.png)

> 在上面的图例中，每一条黑线传输着一整个向量，从一个节点的输出到其他节点的输入。粉色的圈代表按位 pointwise 的操作，诸如向量的和，而黄色的矩阵就是学习到的神经网络层。合在一起的线表示向量的连接，分开的线表示内容被复制，然后分发到不同的位置。

> LSTM 的关键就是细胞状态cell state，水平线在图上方贯穿运行，也就是贯穿每个重复结构的上面这条flow。细胞状态类似于传送带，直接在整个链上运行，只有一些少量的线性交互。信息在上面流传保持不变会很容易。这条flow其实就承载着之前所有状态的信息，每当flow流经一个重复结构A的时候，都会有相应的操作来决定舍弃什么旧的信息以及添加什么新的信息。

![image](https://github.com/ShaoQiBNU/RNN/blob/master/images/8.png)

> LSTM 有通过精心设计对信息增减进行控制的结构，称作为“门”。门是一种让信息选择式通过的方法。他们包含一个 sigmoid 神经网络层和一个按位的乘法操作。Sigmoid 层输出 0 到 1 之间的数值，描述每个部分有多少量可以通过。0 代表“不许任何量通过”，1 就指“允许任意量通过”！

![image](https://github.com/ShaoQiBNU/RNN/blob/master/images/9.png)

> LSTM 拥有三个门，来保护和控制细胞状态，分别是遗忘门 (forget gate)、输入门 (input gate)、输出门 (output gate)。下面对这三个门进行详细讲解

## 遗忘门 (forget gate)

> 遗忘门决定了要从cell state中舍弃什么信息。其通过输入上一状态的输出ht-1、当前状态输入信息Xt到一个Sigmoid函数中，产生一个介于0到1之间的数值，与cell state相乘之后来确定舍弃（保留）多少信息。0 表示“完全舍弃”，1 表示“完全保留”。

![image](https://github.com/ShaoQiBNU/RNN/blob/master/images/10.png)

> 上式中，<img src="https://latex.codecogs.com/svg.latex?\dpi{300}&space;W_{f}" title="W_{f}" />
是遗忘门的权重矩阵，<img src="https://latex.codecogs.com/svg.latex?\dpi{300}&space;[h_{t-1},x_{t}]" title="[h_{t-1},x_{t}]" />表示把两个向量连接成一个更长的向量，<img src="https://latex.codecogs.com/svg.latex?\dpi{300}&space;b_{f}" title="b_{f}" />是遗忘门的偏置项，<img src="https://latex.codecogs.com/svg.latex?\dpi{300}&space;\sigma" title="\sigma" />是sigmoid函数。如果输入的维度是<img src="https://latex.codecogs.com/svg.latex?\dpi{300}&space;d_{x}" title="d_{x}" />，隐藏层的维度是<img src="https://latex.codecogs.com/svg.latex?\dpi{300}&space;d_{h}" title="d_{h}" />，单元状态的维度是<img src="https://latex.codecogs.com/svg.latex?\dpi{300}&space;d_{c}" title="d_{c}" /> （通常 <img src="https://latex.codecogs.com/svg.latex?\dpi{300}&space;d_{c}=d_{h}" title="\d_{c}=d_{h}" />），则遗忘门的权重矩阵<img src="https://latex.codecogs.com/svg.latex?\dpi{300}&space;W_{f}" title="W_{f}" />维度是<img src="https://latex.codecogs.com/svg.latex?\dpi{300}&space;d_{c}&space;\times&space;(d_{h}&plus;d_{x})" title="d_{c} \times (d_{h}+d_{x})" />。事实上，权重矩阵<img src="https://latex.codecogs.com/svg.latex?\dpi{300}&space;W_{f}" title="W_{f}" />都是两个矩阵拼接而成的：一个是<img src="https://latex.codecogs.com/svg.latex?\dpi{300}&space;W_{fh}" title="W_{fh}" />，它对应着输入项<img src="https://latex.codecogs.com/svg.latex?\dpi{300}&space;h_{t-1}" title="h_{t-1}" />，其维度为<img src="https://latex.codecogs.com/svg.latex?\dpi{300}&space;d_{c}\times&space;d_{h}" title="d_{c}\times d_{h}" />；一个是<img src="https://latex.codecogs.com/svg.latex?\dpi{300}&space;W_{fx}" title="W_{fx}" />，它对应着输入项<img src="https://latex.codecogs.com/svg.latex?\dpi{300}&space;x_{t}" title="x_{t}" />，其维度为<img src="https://latex.codecogs.com/svg.latex?\dpi{300}&space;d_{c}\times&space;d_{x}" title="d_{c}\times d_{x}" />。<img src="https://latex.codecogs.com/svg.latex?\dpi{300}&space;W_{f}" title="W_{f}" />可以写为：
<img src="https://latex.codecogs.com/svg.latex?\dpi{300}&space;[W_{f}]\begin{bmatrix}&space;h_{t-1}\\&space;x_{t}&space;\end{bmatrix}=\begin{bmatrix}&space;W_{fh}&space;&&space;W_{fx}&space;\end{bmatrix}\begin{bmatrix}&space;h_{t-1}\\&space;x_{t}&space;\end{bmatrix}=W_{fh}h_{t-1}&plus;W_{fx}x_{t}" title="[W_{f}]\begin{bmatrix} h_{t-1}\\ x_{t} \end{bmatrix}=\begin{bmatrix} W_{fh} & W_{fx} \end{bmatrix}\begin{bmatrix} h_{t-1}\\ x_{t} \end{bmatrix}=W_{fh}h_{t-1}+W_{fx}x_{t}" />


## 输入门 (input gate)

> 输入门决定了要往cell state中保存什么新的信息。其通过输入上一状态的输出<img src="https://latex.codecogs.com/svg.latex?\dpi{300}&space;h_{t-1}" title="h_{t-1}" />、当前状态输入信息<img src="https://latex.codecogs.com/svg.latex?\dpi{300}&space;x_{t}" title="x_{t}" />到一个Sigmoid函数中，产生一个介于0到1之间的数值<img src="https://latex.codecogs.com/svg.latex?\dpi{300}&space;i_{t}" title="i_{t}" />来确定我们需要保留多少的新信息。同时，一个tanh层会通过上一状态的输出<img src="https://latex.codecogs.com/svg.latex?\dpi{300}&space;h_{t-1}" title="h_{t-1}" />、当前状态输入信息<img src="https://latex.codecogs.com/svg.latex?\dpi{300}&space;x_{t}" title="x_{t}" />来得到一个将要加入到cell state中的“候选新信息”<img src="https://latex.codecogs.com/svg.latex?\dpi{300}&space;\widetilde{C_{t}}" title="\widetilde{C_{t}}" />。

![image](https://github.com/ShaoQiBNU/RNN/blob/master/images/11.png)

现在计算当前时刻的单元状态。它是由上一次的单元状态按元素乘以遗忘门，丢弃掉我们确定需要丢弃的信息；然后把当前输入的单元状态按元素乘以输入门，将两个积加和，这就是新的候选值：

<img src="https://latex.codecogs.com/svg.latex?\dpi{300}&space;C_{t}=f_{t}*C_{t-1}&plus;i_{t}*\widetilde{C_{t}}" title="C_{t}=f_{t}*C_{t-1}+i_{t}*\widetilde{C_{t}}" />

![image](https://github.com/ShaoQiBNU/RNN/blob/master/images/12.png)

## 输出门 (output gate)

> 输出门决定了要从cell state中输出什么信息。这个输出将会基于我们的细胞状态，但是也是一个过滤后的版本，会先有一个Sigmoid函数产生一个介于0到1之间的数值<img src="https://latex.codecogs.com/svg.latex?\dpi{300}&space;o_{t}" title="o_{t}" />来确定我们需要输出多少cell state中的信息。cell state的信息再与<img src="https://latex.codecogs.com/svg.latex?\dpi{300}&space;o_{t}" title="o_{t}" />相乘时首先会经过一个tanh层进行“激活”（非线性变换）。得到的就是这个LSTM block的输出信息<img src="https://latex.codecogs.com/svg.latex?\dpi{300}&space;h_{t}" title="h_{t}" />。

![image](https://github.com/ShaoQiBNU/RNN/blob/master/images/13.png)

# 六. 代码

> 采用单层LSTM实现MNIST分类判别，MNIST的输入为影像，影像的行排列需要有一定的顺序，如果胡乱排列，则无法判断数字，因此可以将此问题看作是RNN。设置时间序列长度为28，每次输入影像的一行（28个维度），batch size为128，隐层结点数为128，代码如下：

```python
########## load packages ##########
import tensorflow as tf

##################### load data ##########################
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("mnist_sets",one_hot=True)

########## set net hyperparameters ##########
learning_rate=0.001

epochs=1
batch_size=128

display_step=20

########## set net parameters ##########
n_inputs = 28   # 输入向量的维度，每个时刻的输入特征是28维的，就是每个时刻输入一行，一行有 28 个像素
n_steps = 28    # 循环层长度，即时序持续长度为28，即每做一次预测，需要先输入28行

#### 0-9 digits ####
n_classes=10

#### neurons in hidden layer 隐含层的结点数 ####
n_hidden_units=128

########## placeholder ##########
x=tf.placeholder(tf.float32,[None, n_steps, n_inputs])
y=tf.placeholder(tf.float32,[None, n_classes])


######### Define weights and biases #########
# in:每个cell输入的全连接层参数
# out:定义用于输出的全连接层参数
weights = {
    # (28, 128)
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    # (128, 10)
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}
biases = {
    # (128, )
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    # (10, )
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
}

##################### build net model ##########################
##### RNN LSTM 单层LSTM #######
def RNN(x, weights, biases):

    # hidden layer for input to cell
    # x (128 batch,28 steps,28 inputs) ==> (128 batch * 28 steps, 28 inputs)
    x=tf.reshape(x,shape=[-1, n_inputs])
    
    # into hidden
    # x_in =[128 bach*28 steps,28 inputs]*[28 inputs,128 hidden_units]=[128 batch * 28 steps, 128 hidden]
    x_in = tf.matmul(x, weights['in']) + biases['in']
    
    # x_in ==> (128 batch, 28 steps, 128 hidden)
    x_in = tf.reshape(x_in, [-1, n_steps, n_hidden_units])
 
 
    # cell
    # basic LSTM Cell.初始的forget_bias=1,不希望遗忘任何信息
    cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units,forget_bias=1.0,state_is_tuple=True)
 

    # lstm cell is divided into two parts (c_state, h_state)
    init_state = cell.zero_state(batch_size, dtype=tf.float32)
 

    # dynamic_rnn receive Tensor (batch, steps, inputs) or (steps, batch, inputs) as x_in.
    # n_steps位于次要维度 time_major=False   outputs shape 128, 28, 128
    outputs, final_state = tf.nn.dynamic_rnn(cell, x_in, initial_state=init_state, time_major=False)

 
    # hidden layer for output as the final results
    # unpack to list [(batch, outputs)..] * steps   
    # steps即时间序列长度，此时输出28个ht，由于输入的是 batch steps inputs，因此需要对outputs做调整，从而取到最后一个ht
    # permute time_step_size and batch_size  outputs shape from [128, 28, 128] to [28, 128, 128]
    outputs = tf.unstack(tf.transpose(outputs, [1,0,2]))


    #选择最后一个output与输出的全连接weights相乘再加上biases
    results = tf.matmul(outputs[-1], weights['out']) + biases['out']    # shape = (128, 10)

    return results

########## define model, loss and optimizer ##########

#### model pred 影像判断结果 ####
pred=RNN(x,weights,biases)

#### loss 损失计算 ####
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

#### optimization 优化 ####
optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

#### accuracy 准确率 ####
correct_pred=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))


##################### train and evaluate model ##########################

########## initialize variables ##########
init=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    step=1

    #### epoch 世代循环 ####
    for epoch in range(epochs+1):

        #### iteration ####
        for _ in range(mnist.train.num_examples//batch_size):

            step += 1

            ##### get x,y #####
            batch_x, batch_y=mnist.train.next_batch(batch_size)

            batch_x = batch_x.reshape([batch_size, n_steps, n_inputs])

            ##### optimizer ####
            sess.run(optimizer,feed_dict={x:batch_x, y:batch_y})

            
            ##### show loss and acc ##### 
            if step % display_step==0:
                loss,acc=sess.run([cost, accuracy],feed_dict={x: batch_x, y: batch_y})
                print("Epoch "+ str(epoch) + ", Minibatch Loss=" + \
                    "{:.6f}".format(loss) + ", Training Accuracy= "+ \
                    "{:.5f}".format(acc))


    print("Optimizer Finished!")

    ##### test accuracy #####
    for _ in range(mnist.test.num_examples//batch_size):
        batch_x,batch_y=mnist.test.next_batch(batch_size)
        batch_x = batch_x.reshape([batch_size, n_steps, n_inputs])
        print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: batch_x, y: batch_y}))
```

# 七. LSTM文本分类
> LSTM由于其设计的特点，非常适合用于对时序数据的建模，如文本数据。将词的表示组合成句子的表示，可以采用相加的方法，即将所有词的表示进行加和，或者取平均等方法，但是这些方法没有考虑到词语在句子中前后顺序。如句子“我不觉得他好”。“不”字是对后面“好”的否定，即该句子的情感极性是贬义。使用LSTM模型可以更好的捕捉到较长距离的依赖关系。因为LSTM通过训练过程可以学到记忆哪些信息和遗忘哪些信息。
>
> 基于Keras框架，采用LSTM实现文本分类。文本采用imdb影评分类语料，共25,000条影评，label标记为正面/负面两种评价。影评已被预处理为词下标构成的序列。方便起见，单词的下标基于它在数据集中出现的频率标定，例如整数3所编码的词为数据集中第3常出现的词。这样的组织方法使得用户可以快速完成诸如“只考虑最常出现的10,000个词，但不考虑最常出现的20个词”这样的操作。词向量没有采用预训练好的向量，训练中生成，采用的网络结构如图所示：

![image](https://github.com/ShaoQiBNU/RNN/blob/master/images/14.png)

> 具体代码如下：

```python
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
```

> 运行结果如下：

```
(25000,)
[1, 14, 22, 16, 43]
(25000,)
1
(25000, 20)
[ 65  16  38   2  88  12  16 283   5  16   2 113 103  32  15  16   2  19
 178  32]
(25000, 2)
[0. 1.]
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_33 (Embedding)     (None, 20, 8)             8000      
_________________________________________________________________
lstm_27 (LSTM)               (None, 20, 256)           271360    
_________________________________________________________________
dropout_5 (Dropout)          (None, 20, 256)           0         
_________________________________________________________________
lstm_28 (LSTM)               (None, 256)               525312    
_________________________________________________________________
dropout_6 (Dropout)          (None, 256)               0         
_________________________________________________________________
dense_27 (Dense)             (None, 2)                 514       
=================================================================
Total params: 805,186
Trainable params: 805,186
Non-trainable params: 0
_________________________________________________________________
None
Train on 25000 samples, validate on 25000 samples
Epoch 1/20
25000/25000 [==============================] - 47s 2ms/step - loss: 0.6618 - acc: 0.5817 - val_loss: 0.5914 - val_acc: 0.6820
Epoch 2/20
25000/25000 [==============================] - 40s 2ms/step - loss: 0.5493 - acc: 0.7170 - val_loss: 0.5316 - val_acc: 0.7281
Epoch 3/20
25000/25000 [==============================] - 41s 2ms/step - loss: 0.5085 - acc: 0.7484 - val_loss: 0.5245 - val_acc: 0.7322
Epoch 4/20
25000/25000 [==============================] - 40s 2ms/step - loss: 0.5012 - acc: 0.7548 - val_loss: 0.5160 - val_acc: 0.7381
Epoch 5/20
25000/25000 [==============================] - 40s 2ms/step - loss: 0.4946 - acc: 0.7559 - val_loss: 0.5165 - val_acc: 0.7384
Epoch 6/20
25000/25000 [==============================] - 41s 2ms/step - loss: 0.4924 - acc: 0.7577 - val_loss: 0.5166 - val_acc: 0.7388
Epoch 7/20
25000/25000 [==============================] - 40s 2ms/step - loss: 0.4867 - acc: 0.7596 - val_loss: 0.5264 - val_acc: 0.7292
Epoch 8/20
25000/25000 [==============================] - 40s 2ms/step - loss: 0.4851 - acc: 0.7614 - val_loss: 0.5262 - val_acc: 0.7400
Epoch 9/20
25000/25000 [==============================] - 40s 2ms/step - loss: 0.4803 - acc: 0.7643 - val_loss: 0.5250 - val_acc: 0.7392
Epoch 10/20
25000/25000 [==============================] - 40s 2ms/step - loss: 0.4774 - acc: 0.7651 - val_loss: 0.5220 - val_acc: 0.7376
Epoch 11/20
25000/25000 [==============================] - 40s 2ms/step - loss: 0.4729 - acc: 0.7696 - val_loss: 0.5225 - val_acc: 0.7365
Epoch 12/20
25000/25000 [==============================] - 39s 2ms/step - loss: 0.4704 - acc: 0.7698 - val_loss: 0.5279 - val_acc: 0.7385
Epoch 13/20
25000/25000 [==============================] - 46s 2ms/step - loss: 0.4682 - acc: 0.7713 - val_loss: 0.5303 - val_acc: 0.7343
Epoch 14/20
25000/25000 [==============================] - 44s 2ms/step - loss: 0.4683 - acc: 0.7729 - val_loss: 0.5297 - val_acc: 0.7325
Epoch 15/20
25000/25000 [==============================] - 43s 2ms/step - loss: 0.4659 - acc: 0.7739 - val_loss: 0.5402 - val_acc: 0.7331
Epoch 16/20
25000/25000 [==============================] - 41s 2ms/step - loss: 0.4587 - acc: 0.7759 - val_loss: 0.5350 - val_acc: 0.7312
Epoch 17/20
25000/25000 [==============================] - 42s 2ms/step - loss: 0.4577 - acc: 0.7771 - val_loss: 0.5488 - val_acc: 0.7334
Epoch 18/20
25000/25000 [==============================] - 42s 2ms/step - loss: 0.4524 - acc: 0.7796 - val_loss: 0.5356 - val_acc: 0.7284
Epoch 19/20
25000/25000 [==============================] - 40s 2ms/step - loss: 0.4492 - acc: 0.7829 - val_loss: 0.5357 - val_acc: 0.7332
Epoch 20/20
25000/25000 [==============================] - 40s 2ms/step - loss: 0.4472 - acc: 0.7840 - val_loss: 0.5532 - val_acc: 0.7216
```

# 八. 双向RNN

## (一) 概念

> 但是利用LSTM对句子进行建模还存在一个问题：无法编码从后到前的信息。在更细粒度的分类时，如对于强程度的褒义、弱程度的褒义、中性、弱程度的贬义、强程度的贬义的五分类任务需要注意情感词、程度词、否定词之间的交互。举一个例子，“这个餐厅脏得不行，没有隔壁好”，这里的“不行”是对“脏”的程度的一种修饰，通过BiLSTM可以更好的捕捉双向的语义依赖。BiLSTM是Bi-directional Long Short-Term Memory的缩写，是由前向LSTM与后向LSTM组合而成。比如，我们对“我爱中国”这句话进行编码，模型如图所示：

Img15

> 前向<img src="https://latex.codecogs.com/svg.latex?LSTM_{L}" title="LSTM_{L}" />依次输入“我”，“爱”，“中国”得到三个向量<img src="https://latex.codecogs.com/svg.latex?\{h_{L0},&space;h_{L1},&space;h_{L2}\}" title="\{h_{L0}, h_{L1}, h_{L2}\}" />，后向<img src="https://latex.codecogs.com/svg.latex?LSTM_{R}" title="LSTM_{R}" />依次输入“中国”，“爱”，“我”得到三个向量<img src="https://latex.codecogs.com/svg.latex?\{h_{R0},&space;h_{R1},&space;h_{R2}\}" title="\{h_{R0}, h_{R1}, h_{R2}\}" />。最后将前向和后向的隐向量进行拼接得到<img src="https://latex.codecogs.com/svg.latex?\{[h_{L0},&space;h_{R2}],&space;[h_{L1},&space;h_{R1}],&space;[h_{L2},&space;h_{R0}]\}" title="\{[h_{L0}, h_{R2}], [h_{L1}, h_{R1}], [h_{L2}, h_{R0}]\}" />，即<img src="https://latex.codecogs.com/svg.latex?\{h_{0},&space;h_{1},&space;h_{2}\}" title="\{h_{0}, h_{1}, h_{2}\}" />。对于情感分类任务来说，采用的句子的表示往往是<img src="https://latex.codecogs.com/svg.latex?[h_{L0},&space;h_{R2}]" title="[h_{L0}, h_{R2}]" />，因为其包含了前向与后向的所有信息，如图所示：

Img16

## (二) Tensorflow的Bi-RNN实现

### 1. tensorflow的Bi-RNN代码

> tensorflow的Bi-RNN代码如下：

```python
def bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, sequence_length=None,
                              initial_state_fw=None, initial_state_bw=None,
                              dtype=None, parallel_iterations=None,
                              swap_memory=False, time_major=False, scope=None):
  if not _like_rnncell(cell_fw):
    raise TypeError("cell_fw must be an instance of RNNCell")
  if not _like_rnncell(cell_bw):
    raise TypeError("cell_bw must be an instance of RNNCell")

  with vs.variable_scope(scope or "bidirectional_rnn"):
    # Forward direction
    with vs.variable_scope("fw") as fw_scope:
      output_fw, output_state_fw = dynamic_rnn(
          cell=cell_fw, inputs=inputs, sequence_length=sequence_length,
          initial_state=initial_state_fw, dtype=dtype,
          parallel_iterations=parallel_iterations, swap_memory=swap_memory,
          time_major=time_major, scope=fw_scope)

    # Backward direction
    if not time_major:
      time_dim = 1
      batch_dim = 0
    else:
      time_dim = 0
      batch_dim = 1

    def _reverse(input_, seq_lengths, seq_dim, batch_dim):
      if seq_lengths is not None:
        return array_ops.reverse_sequence(
            input=input_, seq_lengths=seq_lengths,
            seq_dim=seq_dim, batch_dim=batch_dim)
      else:
        return array_ops.reverse(input_, axis=[seq_dim])

    with vs.variable_scope("bw") as bw_scope:
      inputs_reverse = _reverse(
          inputs, seq_lengths=sequence_length,
          seq_dim=time_dim, batch_dim=batch_dim)
      tmp, output_state_bw = dynamic_rnn(
          cell=cell_bw, inputs=inputs_reverse, sequence_length=sequence_length,
          initial_state=initial_state_bw, dtype=dtype,
          parallel_iterations=parallel_iterations, swap_memory=swap_memory,
          time_major=time_major, scope=bw_scope)

  output_bw = _reverse(
      tmp, seq_lengths=sequence_length,
      seq_dim=time_dim, batch_dim=batch_dim)

  outputs = (output_fw, output_bw)
  output_states = (output_state_fw, output_state_bw)

  return (outputs, output_states)
```

### 2. 代码解读

#### (1) 前向输入

> 首先是对输入数据inputs，调用dynamic_rnn从前往后跑一下，得到output_fw和output_state_fw，其中output_fw是所有inputs的LSTM输出状态，output_state_fw是最终的输出状态，

```python
with vs.variable_scope(scope or "bidirectional_rnn"):
    # Forward direction
    with vs.variable_scope("fw") as fw_scope:
      output_fw, output_state_fw = dynamic_rnn(
          cell=cell_fw, inputs=inputs, sequence_length=sequence_length,
          initial_state=initial_state_fw, dtype=dtype,
          parallel_iterations=parallel_iterations, swap_memory=swap_memory,
          time_major=time_major, scope=fw_scope)
```

#### (2) 反向输入

> 定义一个局部函数：把输入的input_ 按照长度为seq_lengths 调用array_ops.rerverse_sequence 做一次转置：

```python
def _reverse(input_, seq_lengths, seq_dim, batch_dim):
      if seq_lengths is not None:
        return array_ops.reverse_sequence(
            input=input_, seq_lengths=seq_lengths,
            seq_dim=seq_dim, batch_dim=batch_dim)
      else:
        return array_ops.reverse(input_, axis=[seq_dim])
```

> 之后把inputs转置成inputs_reverse，然后对这个inputs_reverse跑一下dynamic_rnn得到tmp和output_state_bw：

```python
with vs.variable_scope("bw") as bw_scope:
      inputs_reverse = _reverse(
          inputs, seq_lengths=sequence_length,
          seq_dim=time_dim, batch_dim=batch_dim)
      tmp, output_state_bw = dynamic_rnn(
          cell=cell_bw, inputs=inputs_reverse, sequence_length=sequence_length,
          initial_state=initial_state_bw, dtype=dtype,
          parallel_iterations=parallel_iterations, swap_memory=swap_memory,
          time_major=time_major, scope=bw_scope)
```

> 再把这个输出tmp反转一下得到Output_bw向量：

```python
output_bw = _reverse(
      tmp, seq_lengths=sequence_length,
      seq_dim=time_dim, batch_dim=batch_dim)
```

#### (3) 前向和反向的LSTM输出堆叠

> output_fw和output_bw堆叠在一起得到bi-rnn的输出，隐藏层状态output_state_fw和output_state_bw堆叠在一起得到bi-rnn的隐藏层状态，最终输出：

```python
  outputs = (output_fw, output_bw)
  output_states = (output_state_fw, output_state_bw)

  return (outputs, output_states)
```

## (三) 代码

> 基于Keras框架，采用双向LSTM实现文本分类，代码如下：

```python
###################### load packages ####################
from keras.datasets import imdb
from keras import preprocessing
from keras.models import Sequential
from keras.layers import Dense, Embedding, Dropout, LSTM, Bidirectional, SpatialDropout1D
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

#### dropout ####
model.add(SpatialDropout1D(0.3))

#### bi-RNN ####
model.add(Bidirectional(LSTM(100, dropout=0.3, recurrent_dropout=0.3)))

#### dense ####
model.add(Dense(1024, activation='relu'))

#### dropout ####
model.add(Dropout(0.8))

#### dense ####
model.add(Dense(1024, activation='relu'))

#### dropout ####
model.add(Dropout(0.8))

#### 输出层 ####
model.add(Dense(num_class, activation='softmax'))

print(model.summary())

######## optimization and train ########
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, batch_size=512, epochs=20, verbose=1, validation_data=(x_test, y_test))
```

> 结果如下：

```
(25000,)
[1, 14, 22, 16, 43]
(25000,)
1
(25000, 20)
[ 65  16  38   2  88  12  16 283   5  16   2 113 103  32  15  16   2  19
 178  32]
(25000, 2)
[0. 1.]
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_1 (Embedding)      (None, 20, 8)             8000      
_________________________________________________________________
spatial_dropout1d_1 (Spatial (None, 20, 8)             0         
_________________________________________________________________
bidirectional_1 (Bidirection (None, 200)               87200     
_________________________________________________________________
dense_1 (Dense)              (None, 1024)              205824    
_________________________________________________________________
dropout_1 (Dropout)          (None, 1024)              0         
_________________________________________________________________
dense_2 (Dense)              (None, 1024)              1049600   
_________________________________________________________________
dropout_2 (Dropout)          (None, 1024)              0         
_________________________________________________________________
dense_3 (Dense)              (None, 2)                 2050      
=================================================================
Total params: 1,352,674
Trainable params: 1,352,674
Non-trainable params: 0
_________________________________________________________________
None
Train on 25000 samples, validate on 25000 samples
Epoch 1/20
25000/25000 [==============================] - 22s 884us/step - loss: 0.6930 - acc: 0.5094 - val_loss: 0.6887 - val_acc: 0.5946
Epoch 2/20
25000/25000 [==============================] - 18s 700us/step - loss: 0.6251 - acc: 0.6475 - val_loss: 0.5463 - val_acc: 0.7220
Epoch 3/20
25000/25000 [==============================] - 18s 703us/step - loss: 0.5466 - acc: 0.7254 - val_loss: 0.5231 - val_acc: 0.7366
Epoch 4/20
25000/25000 [==============================] - 18s 704us/step - loss: 0.5296 - acc: 0.7377 - val_loss: 0.5179 - val_acc: 0.7367
Epoch 5/20
25000/25000 [==============================] - 18s 700us/step - loss: 0.5209 - acc: 0.7432 - val_loss: 0.5207 - val_acc: 0.7310
Epoch 6/20
25000/25000 [==============================] - 18s 702us/step - loss: 0.5151 - acc: 0.7452 - val_loss: 0.5144 - val_acc: 0.7380
Epoch 7/20
25000/25000 [==============================] - 17s 694us/step - loss: 0.5118 - acc: 0.7488 - val_loss: 0.5123 - val_acc: 0.7390
Epoch 8/20
25000/25000 [==============================] - 18s 727us/step - loss: 0.5064 - acc: 0.7542 - val_loss: 0.5153 - val_acc: 0.7361
Epoch 9/20
25000/25000 [==============================] - 18s 708us/step - loss: 0.5060 - acc: 0.7540 - val_loss: 0.5119 - val_acc: 0.7400
Epoch 10/20
25000/25000 [==============================] - 18s 720us/step - loss: 0.5042 - acc: 0.7518 - val_loss: 0.5110 - val_acc: 0.7401
Epoch 11/20
25000/25000 [==============================] - 18s 731us/step - loss: 0.5052 - acc: 0.7508 - val_loss: 0.5126 - val_acc: 0.7415
Epoch 12/20
25000/25000 [==============================] - 18s 736us/step - loss: 0.5003 - acc: 0.7578 - val_loss: 0.5114 - val_acc: 0.7400
Epoch 13/20
25000/25000 [==============================] - 19s 741us/step - loss: 0.4983 - acc: 0.7554 - val_loss: 0.5164 - val_acc: 0.7362
Epoch 14/20
25000/25000 [==============================] - 23s 925us/step - loss: 0.4976 - acc: 0.7616 - val_loss: 0.5115 - val_acc: 0.7403
Epoch 15/20
25000/25000 [==============================] - 23s 926us/step - loss: 0.4949 - acc: 0.7599 - val_loss: 0.5118 - val_acc: 0.7401
Epoch 16/20
25000/25000 [==============================] - 23s 903us/step - loss: 0.4957 - acc: 0.7608 - val_loss: 0.5110 - val_acc: 0.7403
Epoch 17/20
25000/25000 [==============================] - 23s 926us/step - loss: 0.4919 - acc: 0.7610 - val_loss: 0.5109 - val_acc: 0.7405
Epoch 18/20
25000/25000 [==============================] - 23s 927us/step - loss: 0.4909 - acc: 0.7622 - val_loss: 0.5107 - val_acc: 0.7408
Epoch 19/20
25000/25000 [==============================] - 23s 921us/step - loss: 0.4886 - acc: 0.7608 - val_loss: 0.5108 - val_acc: 0.7397
Epoch 20/20
25000/25000 [==============================] - 23s 917us/step - loss: 0.4895 - acc: 0.7624 - val_loss: 0.5132 - val_acc: 0.7369
```


参考：

http://colah.github.io/posts/2015-08-Understanding-LSTMs/

https://www.jianshu.com/p/9dc9f41f0b29/

https://blog.csdn.net/jmh1996/article/details/83476061



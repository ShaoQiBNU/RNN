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

参考：

http://colah.github.io/posts/2015-08-Understanding-LSTMs/.

https://www.jianshu.com/p/9dc9f41f0b29/.



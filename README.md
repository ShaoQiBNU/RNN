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

> 上式中， 是遗忘门的权重矩阵，表示把两个向量连接成一个更长的向量，是遗忘门的偏置项，是sigmoid函数。如果输入的维度是，隐藏层的维度是，单元状态的维度是（通常），则遗忘门的权重矩阵维度是。事实上，权重矩阵都是两个矩阵拼接而成的：一个是，它对应着输入项，其维度为；一个是，它对应着输入项，其维度为。可以写为：

## 输入门 (input gate)

## 输出门 (output gate)

<img src="https://latex.codecogs.com/svg.latex?\dpi{300}&space;ax^{2}&space;&plus;&space;by^{2}&space;&plus;&space;c&space;=&space;0" title="ax^{2} + by^{2} + c = 0" />

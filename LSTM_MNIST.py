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
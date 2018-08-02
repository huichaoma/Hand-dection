import training
import tensorflow as tf
import numpy as np
import random
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#预测标签
output=['Call','Eight','Fight','Good',
        'I','Love','Nine','Ok','Stop','V']

global_step=0
batch_size = 40
INPUT_NODE = training.img_rows * training.img_cols
OUTPUT_NODE = 10

Image_size = 200
NUM_LABELS = 10

# 第一层卷积层的尺寸和深度
CONV1_DEEP = 32
CONV1_SIZE= 5

# 第二层卷积层的尺寸和深度
CONV2_DEEP = 64
CONV2_SIZE= 3

# 第三层卷积层的尺寸和深度
CONV3_DEEP = 64
CONV3_SIZE= 5

# 第四层卷积层的尺寸和深度
CONV4_DEEP = 64
CONV4_SIZE= 5

FC_SIZE1 = 512
FC_SIZE2 = 128

# 训练用参数
ACCURACY=0.998
REGULARIZATION_RATE = 0.0002
TRAINING_STEPS = 30000
MODEL_SAVE_PATH = 'D:\Interns software\code\detection\model'
MODEL_NAME = 'model.ckpt'



def get_batch(X, y, batch_size):
    data = []
    label = []
    m = X.shape[0]
    for _ in range(batch_size):
        index = random.randrange(m)
        data.append(X[index])
        tmp = np.zeros(NUM_LABELS, dtype=np.float32)
        tmp[y[index]] = 1.0
        label.append(tmp)
    return np.array(data), np.array(label)


# 定义前向卷积 添加：dropout 训练有 测试没有
def inference(input_tensor, train, regularizer):
    with tf.name_scope('layer1-conv1'):
        conv1_weights = tf.get_variable("weight", [CONV1_SIZE, CONV1_SIZE, training.img_channels, CONV1_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable("bias", [CONV1_DEEP], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding="VALID") # 196*196*32
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    with tf.name_scope('layer2-pool1'):
        pool1 = tf.nn.max_pool(relu1, ksize=[1,2,2,1], strides=[1, 2, 2, 1], padding="SAME") # 98*98*32

    with tf.variable_scope('layer3-conv2'):
        conv2_weight = tf.get_variable("weight", [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable('bias', [CONV2_DEEP], initializer=tf.constant_initializer(0.0))

        conv2 = tf.nn.conv2d(pool1, conv2_weight, strides=[1,1,1,1], padding="VALID") # 96*96*64
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    with tf.name_scope('layer4-pool2'):
        pool2 = tf.nn.max_pool(relu2, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME") # 48*48*64

    with tf.variable_scope('layer5-conv3'):
        conv3_weight = tf.get_variable("weight", [CONV3_SIZE, CONV3_SIZE, CONV2_DEEP, CONV3_DEEP],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv3_biases = tf.get_variable('bias', [CONV3_DEEP], initializer=tf.constant_initializer(0.0))

        conv3 = tf.nn.conv2d(pool2, conv3_weight, strides=[1,1,1,1], padding="VALID") # 44*44*64
        relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_biases))

    with tf.name_scope('layer6-pool3'):
        pool3 = tf.nn.max_pool(relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")  # 22*22*64

    with tf.variable_scope('layer7-conv4'):
        conv4_weight = tf.get_variable("weight", [CONV4_SIZE, CONV4_SIZE, CONV3_DEEP, CONV4_DEEP],
                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv4_biases = tf.get_variable('bias', [CONV4_DEEP], initializer=tf.constant_initializer(0.0))

        conv4 = tf.nn.conv2d(pool3, conv4_weight, strides=[1, 1, 1, 1], padding="VALID")  # 18*18*64
        relu4 = tf.nn.relu(tf.nn.bias_add(conv4, conv4_biases))

    with tf.name_scope('layer8-pool4'):
        pool4 = tf.nn.max_pool(relu4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")  # 9*9*64

    # 然后将第8层的输出变为第9层输入的格式。 后面全连接层需要输入的是向量 将矩阵拉成一个向量
    pool_shape = pool4.get_shape().as_list()
    # pool_shape[0]为一个batch中数据个数
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    reshaped = tf.reshape(pool4, [pool_shape[0], nodes])

    with tf.variable_scope('layer9-fc1'):
        fc1_weights = tf.get_variable("weight", [nodes, FC_SIZE1],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))

        # 只有全连接层加入正则化
        if regularizer != None:
            tf.add_to_collection('loss', regularizer(fc1_weights))
        fc1_biases = tf.get_variable('bias', [FC_SIZE1], initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        if train:
            fc1 = tf.nn.dropout(fc1, 0.5)

    with tf.variable_scope('layer10-fc2'):
        fc2_weights = tf.get_variable("weight", [FC_SIZE1, FC_SIZE2],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))

        # 只有全连接层加入正则化
        if regularizer != None:
            tf.add_to_collection('loss', regularizer(fc2_weights))
        fc2_biases = tf.get_variable('bias', [FC_SIZE2], initializer=tf.constant_initializer(0.1))
        fc2 = tf.nn.relu(tf.matmul(fc1, fc2_weights) + fc2_biases)
        if train:
            fc2 = tf.nn.dropout(fc2, 0.5)

    with tf.variable_scope('layer11-fc3'):
        fc3_weights = tf.get_variable("weight", [FC_SIZE2, NUM_LABELS],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))

        # 只有全连接层加入正则化
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc3_weights))
        fc3_biases = tf.get_variable('bias', [NUM_LABELS], initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(fc2, fc3_weights) + fc3_biases

    return logit # 注 ： 这里没有经过softmax，后面在计算cross_entropy时候利用内置的函数会计算。
def train(X_train, y_train):
    if(not os.path.exists(MODEL_SAVE_PATH)):
        os.makedirs(MODEL_SAVE_PATH)
    x = tf.placeholder(tf.float32, [batch_size, training.img_rows, training.img_cols, training.img_channels], name='x-input')
    y = tf.placeholder(tf.float32, [batch_size, OUTPUT_NODE], name='y-input')

    # 正则化
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

    # 前向传播
    y_ = inference(x, train=True, regularizer=regularizer) # 预测值
    global_step = tf.Variable(0, trainable=False) # 不可训练

    #定义损失函数
    # 滑动平均
    # variable_averages = tf.train.ExponentialMovingAverage(MOVING_AAVERAGE_DECAY, global_step)
    # variable_averages_op = variable_averages.apply(tf.trainable_variables())

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y, 1), logits=y_) #softmax和交叉熵计算
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses')) # 计算总loss

    # learninig_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, 1204//batch_szie, LEARNING_RATE_DECAY)
    train_step = tf.train.GradientDescentOptimizer(0.0005).minimize(loss, global_step=global_step)#学习率

    # with tf.control_dependencies([train_step, variable_averages_op]):
    #     train_op = tf.no_op(name='train')
    # train_op = tf.group(train_step, variable_averages_op)

    # 保存模型
    saver = tf.train.Saver()
    max_acc = 0
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        for i in range(TRAINING_STEPS):
            xs, ys = get_batch(X_train, y_train, batch_size=batch_size)
            # xs, ys = get_next_batch(X_train, y_train, batch_size=batch_szie)
            # ys = tf.reshape(tf.one_hot(ys, depth=5), [batch_szie, OUTPUT_NODE])
            _, loss_value, step = sess.run([train_step, loss, global_step], feed_dict={x: xs, y: ys})
            print("----------------------", i, " : ", loss_value,  "-------------------------------")
            if (1-loss_value) > max_acc:  # 记录测试准确率最大时的模型
                max_acc =1-loss_value
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME))  # 保存模型。
            if  max_acc>ACCURACY:  # 达到这个准确率跳出训练循环
                break
        return step

def test(X_test, y_test):
    # EVAL_INTERVAL_SECS = 10 # 每10秒加载一次模型，并在测试数据上测试准确率
    with tf.Graph().as_default() as g: # 设置默认graph
        # 定义输入输出格式
        #
        x = tf.placeholder(tf.float32, [1, training.img_rows, training.img_cols, training.img_channels], name='x-input')
        y = tf.placeholder(tf.float32, [1, OUTPUT_NODE], name='y-input')
        #print(x.shape,y.shape)

        y_ = inference(x, train=None, regularizer=None) # 测试时 不关注正则化损失的值

        # 开始计算正确率
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # 加载模型
        saver = tf.train.Saver()
        with tf.Session() as sess:
            # tf.train.get_checkpoint_state会自动找到目录中的最新模型文件名
            ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                # 得到迭代轮数
                for _ in range(X_test.shape[0]):
                    xs, ys = get_batch(X_test, y_test, batch_size=1) # 测试用
                    # print(ys.shape)
                    label, accuracy_score = sess.run([y_, accuracy], feed_dict={x: xs, y: ys})
                    print("实际手势： %s，  预测手势： %s" % (output[np.argmax(ys)], output[np.argmax(label)]))
                    print("After %s training steps(s), test accuracy = %f" % (global_step, accuracy_score))
            else:
                print("No checkpoint, Training Firstly.")
                return
def Gussgesture(X_test):
    # EVAL_INTERVAL_SECS = 10 # 每10秒加载一次模型，并在测试数据上测试准确率
    with tf.Graph().as_default() as g: # 设置默认graph
        # 定义输入输出格式
        #
        x = tf.placeholder(tf.float32, [1, training.img_rows, training.img_cols, training.img_channels], name='x-input')
        y_ = inference(x, train=None, regularizer=None) # 测试时 不关注正则化损失的值

        # 开始计算正确率
        # correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1))
        # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # 加载模型
        saver = tf.train.Saver()
        with tf.Session() as sess:
            # tf.train.get_checkpoint_state会自动找到目录中的最新模型文件名
            ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                # 得到迭代轮数
                # global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1] # model.ckpt-3000
                # print(ys.shape)
                label = sess.run(y_, feed_dict={x: X_test.reshape(1, X_test.shape[0], X_test.shape[1], X_test.shape[2])})
                #print("预测手势： %s" % (output[np.argmax(label)]))
                # PLOT(label)
                # print("After %s training steps(s), test accuracy = %f" % (global_step, accuracy_score))
                return output[np.argmax(label)]
            else:
                print("No checkpoint, Training Firstly.")
                return
if __name__=='__main__':
    a,b,c,d=training.Initializer()
    global_step=train(a,c)
    test(b,d)
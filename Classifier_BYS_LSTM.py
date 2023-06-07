import paddle
import paddle.dataset.imdb as imdb
import paddle.fluid as fluid
import numpy as np
import os

word_dict = imdb.word_dict()
dict_dim = len(word_dict)
# 获取训练和预测数据
train_reader = paddle.batch(paddle.reader.shuffle(imdb.train(word_dict), 512), batch_size=128)
test_reader = paddle.batch(imdb.test(word_dict), batch_size=128)


# 定义长短期记忆网络

def lstm_net(ipt, input_dim):
    # 以数据的IDs作为输入

    emb = fluid.layers.embedding(input=ipt, size=[input_dim, 128], is_sparse=True)

    # 第一个全连接层

    fc1 = fluid.layers.fc(input=emb, size=128)

    # 进行一个长短期记忆操作

    lstm1, _ = fluid.layers.dynamic_lstm(input=fc1,  # 返回：隐藏状态（hidden state），LSTM的神经元状态

                                         size=128)  # size=4*hidden_size

    # 第一个最大序列池操作

    fc2 = fluid.layers.sequence_pool(input=fc1, pool_type='max')

    # 第二个最大序列池操作

    lstm2 = fluid.layers.sequence_pool(input=lstm1, pool_type='max')

    # 以softmax作为全连接的输出层，大小为2,也就是正负面

    out = fluid.layers.fc(input=[fc2, lstm2], size=2, act='softmax')

    return out


# 定义卷积神经网络
def cnn_net(ipt, input_dim):
    # 嵌入层
    emb = fluid.layers.embedding(input=ipt, size=[input_dim, 128], is_sparse=True)

    # 第一个卷积层-池化层
    conv_pool_1 = fluid.nets.sequence_conv_pool(
        input=emb,
        filter_size=3,  # 滤波器的大小
        num_filters=128,  # filter的数量，它与输出的通道相同
        act="tanh",  # 激活类型
        pool_type="max")
    # 第二个卷积层
    conv_pool_2 = fluid.nets.sequence_conv_pool(
        input=emb,
        filter_size=4,  # 滤波器的大小
        num_filters=128,  # filter的数量，它与输出的通道相同
        act="tanh",  # 激活类型
        pool_type="max")

    # 以softmax作为全连接的输出层，大小为2,也就是正负面
    out = fluid.layers.fc(input=[conv_pool_1, conv_pool_2], size=2, act="softmax")

    return out


# 定义输入数据， lod_level不为0指定输入数据为序列数据

words = fluid.layers.data(name='words', shape=[1], dtype='int64', lod_level=1)

label = fluid.layers.data(name='label', shape=[1], dtype='int64')

# 获取长短期记忆网络

model = lstm_net(words, dict_dim)

# 获取卷积神经网络

# model = cnn_net(words, dict_dim)
# 获取损失函数和准确率

cost = fluid.layers.cross_entropy(input=model, label=label)

avg_cost = fluid.layers.mean(cost)

acc = fluid.layers.accuracy(input=model, label=label)
# 获取预测程序

test_program = fluid.default_main_program().clone(for_test=True)
# 定义优化方法

optimizer = fluid.optimizer.AdagradOptimizer(learning_rate=0.002)

opt = optimizer.minimize(avg_cost)
# 定义输入数据的维度

# 定义数据数据的维度，数据的顺序是一条句子数据对应一个标签

feeder = fluid.DataFeeder(place=place, feed_list=[words, label])

# 开始训练

for pass_id in range(1):

    # 进行训练

    train_cost = 0

    for batch_id, data in enumerate(train_reader()):  # 遍历train_reader迭代器

        train_cost = exe.run(program=fluid.default_main_program(),  # 运行主程序

                             feed=feeder.feed(data),  # 喂入一个batch的数据

                             fetch_list=[avg_cost])  # fetch均方误差

        if batch_id % 40 == 0:  # 每40次batch打印一次训练、进行一次测试

            print('Pass:%d, Batch:%d, Cost:%0.5f' % (pass_id, batch_id, train_cost[0]))

    # 进行测试

    test_costs = []  # 测试的损失值

    test_accs = []  # 测试的准确率

    for batch_id, data in enumerate(test_reader()):
        test_cost, test_acc = exe.run(program=test_program,

                                      feed=feeder.feed(data),

                                      fetch_list=[avg_cost, acc])

        test_costs.append(test_cost[0])

        test_accs.append(test_acc[0])

    # 计算平均预测损失在和准确率

    test_cost = (sum(test_costs) / len(test_costs))

    test_acc = (sum(test_accs) / len(test_accs))

    print('Test:%d, Cost:%0.5f, ACC:%0.5f' % (pass_id, test_cost, test_acc))

# 保存模型

model_save_dir = "/home/aistudio/work/emotionclassify.inference.lstmmodel"

# 如果保存路径不存在就创建

if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)

print('save models to %s' % (model_save_dir))

fluid.io.save_inference_model(model_save_dir,  # 保存推理model的路径

                              ['words'],  # 推理（inference）需要 feed 的数据

                              [model],  # 保存推理（inference）结果的 Variables

                              exe)  # exe 保存 inference model

# 定义预测数据

reviews_str = ['read the book forget the movie', 'this is a great movie', 'this is very bad']

# 把每个句子拆成一个个单词

reviews = [c.split() for c in reviews_str]

# 获取结束符号的标签

UNK = word_dict['<unk>']

# 获取每句话对应的标签

lod = []

for c in reviews:
    # 需要把单词进行字符串编码转换

    lod.append([word_dict.get(words.encode('utf-8'), UNK) for words in c])

# 获取每句话的单词数量

base_shape = [[len(c) for c in lod]]

# 生成预测数据

tensor_words = fluid.create_lod_tensor(lod, base_shape, place)

infer_exe = fluid.Executor(place)  # 创建推测用的executor

inference_scope = fluid.core.Scope()  # Scope指定作用域

with fluid.scope_guard(inference_scope):  # 修改全局/默认作用域（scope）, 运行时中的所有变量都将分配给新的scope。

    # 从指定目录中加载 推理model(inference model)

    [inference_program,  # 推理的program

     feed_target_names,  # str列表，包含需要在推理program中提供数据的变量名称

     fetch_targets] = fluid.io.load_inference_model(model_save_dir,  # fetch_targets: 推断结果，model_save_dir:模型训练路径

                                                    infer_exe)  # infer_exe: 运行 inference model的 executor

    results = infer_exe.run(inference_program,  # 运行预测程序

                            feed={feed_target_names[0]: tensor_words},  # 喂入要预测的x值

                            fetch_list=fetch_targets)  # 得到推测结果

    # 打印每句话的正负面概率

    for i, r in enumerate(results[0]):
        print("\'%s\'的预测结果为：正面概率为：%0.5f，负面概率为：%0.5f" % (reviews_str[i], r[0], r[1]))

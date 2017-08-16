import tensorflow as tf
import numpy as np

# 英语翻译成中文

# 词库
seq_raw_vob = [['word', '单词'], ['wood', '木头'],
               ['game', '游戏'], ['girl', '女孩'],
               ['kiss', '接吻'], ['love', '爱情']]

# 构建字典
char_list = []
for ele in seq_raw_vob:
    for c in ele:
        char_list.extend(list(c))

char_list.extend(['S', 'E', 'P'])
char_list.extend(list('abcdefghijklmnopqrstuvwxyz'))

char_list = list(set(char_list))
char_to_int_dict = {c: i for i, c in enumerate(char_list)}

dic_len = len(char_to_int_dict)


# 生成batch形式的训练样本。
def generate_batch(seq_raw_vob):
    input_batch = []
    output_batch = []
    target_batch = []

    for word_pair in seq_raw_vob:

        # 对照字典char_to_int_dict，将输入单词和输出单词数字化
        input = [char_to_int_dict[c] for c in word_pair[0]]
        output = [char_to_int_dict[c] for c in ('S' + word_pair[1])]

        target = [char_to_int_dict[c] for c in (word_pair[1] + 'E')]

        # 利用numpy的eye函数，编码方式为one - hot
        input_batch.append(np.eye(dic_len)[input])
        output_batch.append(np.eye(dic_len)[output])

        target_batch.append(target)
    return input_batch, output_batch, target_batch


# 参数设置
learning_rate = 0.01
lstm_size = 64
epoch_num = 1

# one-hot编码
n_class = n_input = dic_len

# seq2seq模型
# 编码器与解码器的输入格式
# [batch_size, time_step, input_size]
enc_input = tf.placeholder(tf.float32, [None, None, n_input])
dec_input = tf.placeholder(tf.float32, [None, None, n_input])
targets = tf.placeholder(tf.int64, [None, None])

# 编码器
with tf.variable_scope('encode'):
    enc_cell = tf.contrib.rnn.BasicLSTMCell(lstm_size)
    enc_cell = tf.contrib.rnn.DropoutWrapper(enc_cell, output_keep_prob=0.5)

    # enc_outputs: [batch_size, time_step, lstm_size], enc_state: [batch_size, lstm_size]
    enc_outputs, enc_states = tf.nn.dynamic_rnn(enc_cell, enc_input, dtype=tf.float32)
    print('编码过程: enc_outputs is : {}, \n enc_states is {}'.format(enc_outputs, enc_states))

# 解码器
with tf.variable_scope('decode'):
    dec_cell = tf.contrib.rnn.BasicLSTMCell(lstm_size)
    dec_cell = tf.contrib.rnn.DropoutWrapper(dec_cell, output_keep_prob=0.5)

    # 解码器的cell的初始状态是编码器的cell的最后状态
    # dec_outputs:[batch_size, time_step, lstm_size]
    dec_outputs, dec_states = tf.nn.dynamic_rnn(dec_cell, dec_input, initial_state=enc_states, dtype=tf.float32)
    print('解码过程: dec_outputs is : {}, \n dec_states is {}'.format(dec_outputs, dec_states))

# 输出层，size大小为[batch_size, time_step, n_class]
outputs = tf.layers.dense(dec_outputs, n_class, activation=None)

# 计算cost
cost = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=outputs, labels=targets))

# BPTT过程
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)


with tf.Session() as sess:
    init = tf.initialize_all_variables()
    sess.run(init)
    tf.train.start_queue_runners(sess=sess)

    input_batch, output_batch, target_batch = generate_batch(seq_raw_vob)
    print(target_batch)
    for epoch in range(epoch_num):
        enc_outputs_val = tf.unpack(tf.transpose(enc_outputs, [1, 0, 2]))
        enc_outputs_val, enc_state_val = sess.run([enc_outputs, enc_states], feed_dict={enc_input: input_batch})
        _, loss = sess.run([optimizer, cost],
                           feed_dict={enc_input: input_batch,
                                      dec_input: output_batch,
                                      targets: target_batch})
        print('编码过程: enc_outputs is : {},  \n enc_states is {}, \n'.format(enc_outputs_val, enc_state_val[1]))
        print('Epoch:', '%04d' % (epoch + 1),
              'cost =', '{:.6f}'.format(loss))

    print('训练完成!')


    # 测试过程
    def translate(word):
        seq_data = [word, 'P' * len(word)]
        input_batch, output_batch, target_batch = generate_batch([seq_data])
        prediction = tf.argmax(outputs, 2)

        result = sess.run(prediction,
                      feed_dict={enc_input: input_batch,
                                 dec_input: output_batch,
                                 targets: target_batch})
        print(result)
        decoded = [char_list[i] for i in result[0]]
        end = decoded.index('E')
        translated = ''.join(decoded[:end])

        return translated

    print('\n=== 翻译测试 ===')

    print('word ->', translate('word'))
    print('wodr ->', translate('wodr'))
    print('love ->', translate('love'))
    print('loev ->', translate('loev'))
    print('abcd ->', translate('abcd'))
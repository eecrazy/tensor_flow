
# coding: utf-8

# 

# Deep Learning
# =============
# 
# Assignment 6
# ------------
# 
# After training a skip-gram model in `5_word2vec.ipynb`, the goal of this notebook is to train a LSTM character model over [Text8](http://mattmahoney.net/dc/textdata) data.

# ---
# Problem 1
# ---------
# 
# You might have noticed that the definition of the LSTM cell involves 4 matrix multiplications with the input, and 4 matrix multiplications with the output. Simplify the expression by using a single matrix multiply for each, and variables that are 4 times larger.
# 
# LSTM单元涉及跟input的4个矩阵乘积运算，以及跟output的四个矩阵乘积运算。简化这个表达式运算：一个单独的大矩阵乘积运算对于input和output，
# 变量需要4倍大。
# 
# ---

# In[1]:

# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import os
import numpy as np
import random
import string
import tensorflow as tf
import zipfile
from six.moves import range
from six.moves.urllib.request import urlretrieve


# In[2]:

url = 'http://mattmahoney.net/dc/'

def maybe_download(filename, expected_bytes):
  """Download a file if not present, and make sure it's the right size."""
  if not os.path.exists(filename):
    filename, _ = urlretrieve(url + filename, filename)
  statinfo = os.stat(filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified %s' % filename)
  else:
    print(statinfo.st_size)
    raise Exception(
      'Failed to verify ' + filename + '. Can you get to it with a browser?')
  return filename

filename = maybe_download('/users1/zyli/data/text8.zip', 31344016)


# In[3]:

def read_data(filename):
#   这里打开了一个ZIP文件
  f = zipfile.ZipFile(filename)
  for name in f.namelist():
    #这里是tf的读入数据的方法
    return tf.compat.as_str(f.read(name))
  f.close()
  
text = read_data(filename)
print('Data size %d' % len(text))


# Create a small validation set.

# In[4]:

# 构建训练集和验证集
valid_size = 1000
valid_text = text[:valid_size]
train_text = text[valid_size:]
train_size = len(train_text)
print(train_size, train_text[:64])
print(valid_size, valid_text[:64])


# Utility functions to map characters to vocabulary IDs and back.

# In[5]:

vocabulary_size = len(string.ascii_lowercase) + 1 # [a-z] + ' '
# 词表大小共26+1
# 第一个单词是a，ord转换为它的ascii玛值97
first_letter = ord(string.ascii_lowercase[0])

# 将 空格和a-z 映射为 0-26
def char2id(char):
  if char in string.ascii_lowercase:
    return ord(char) - first_letter + 1
  elif char == ' ':
    return 0
  else:
    print('Unexpected character: %s' % char)
    return 0

# 将0-26 映射为 空格和a-z
def id2char(dictid):
  if dictid > 0:
    return chr(dictid + first_letter - 1)
  else:
    return ' '

print(char2id('a'), char2id('z'), char2id(' '), char2id('ï'))
print(id2char(1), id2char(26), id2char(0))


# Function to generate a training batch for the LSTM model.

# In[6]:

# 生成训练的mini_batch，每个batch包含64 * 11个字母
batch_size=64

# 每个训练样本的大小为11
num_unrollings=10


# 这个batch生成函数很神奇！！！值得学习
class BatchGenerator(object):
  def __init__(self, text, batch_size, num_unrollings):
    # 生成batch的文本
    self._text = text
    # 文本总长度
    self._text_size = len(text)
    # batch_size
    self._batch_size = batch_size
    # 每个batch包含的字母个数
    self._num_unrollings = num_unrollings
    # 将整个文本分为64个块，每次取batch的时候分别从64个块里面各取一个，segment表示每个文本块的大小，99999000//64=1562484
    segment = self._text_size // batch_size #整除
    # _cursor表示每个块的起始位置，cursor的大小为64
    self._cursor = [ offset * segment for offset in range(batch_size)]
    
    # 这个是什么意思?
    self._last_batch = self._next_batch()
  
  #从数据中当前的cursor位置生成一个单独的batch
  def _next_batch(self):
    """Generate a single batch from the current cursor position in the data."""
    # 每个batch是一个64*27的矩阵，每行表示一个字母
    batch = np.zeros(shape=(self._batch_size, vocabulary_size), dtype=np.float)
    
    for b in range(self._batch_size):
      batch[b, char2id(self._text[self._cursor[b]])] = 1.0
      #更新当前的cursor位置
      self._cursor[b] = (self._cursor[b] + 1) % self._text_size
    # 生成的batch中包含64个字母，但是这64个字母在文本中不连续
    return batch
  
  def next(self):
    """Generate the next array of batches from the data. The array consists of
    the last batch of the previous array, followed by num_unrollings new ones.
    """
    batches = [self._last_batch]
    for step in range(self._num_unrollings):
      batches.append(self._next_batch())
    self._last_batch = batches[-1]
    
    # batches中包含11个64*27的矩阵
    return batches


# probabilities是一个矩阵，axis=1是往y轴上投影，所以矩阵每一行（27维）表示一个概率分布，
# 返回的列表长度是矩阵的行数，也就是64个字母
def characters(probabilities):
  """Turn a 1-hot encoding or a probability distribution over the possible
  characters back into its (most likely) character representation."""
  return [id2char(c) for c in np.argmax(probabilities, 1)]

# 将好几个batch恢复到它的string形式
def batches2string(batches):
  """Convert a sequence of batches back into their (most likely) string representation."""
  # s先是一个batch的长度，也就是64
  s = [''] * batches[0].shape[0]
  # batches中共有11个64*27的矩阵   
  for b in batches:
    # b是一个batch的概率分布矩阵64*27
    # 下面这个函数太神奇了，可以直接把64*11的字符串拼接在一起
    s = [''.join(x) for x in zip(s, characters(b))]
  return s

# 生成训练和验证的batches
train_batches = BatchGenerator(train_text, batch_size, num_unrollings)
valid_batches = BatchGenerator(valid_text, 1, 1)

# 打印训练和验证的batches
# print(train_batches.next())
print(batches2string(train_batches.next()))
print(batches2string(train_batches.next()))

print(batches2string(valid_batches.next()))
print(batches2string(valid_batches.next()))


# In[7]:

#这个函数干什么用的？
def logprob(predictions, labels):
  """Log-probability of the true labels in a predicted batch."""
  # 把predictions里面小于10的-10次方的数值变为10的-10次方
  predictions[predictions < 1e-10] = 1e-10
  return np.sum(np.multiply(labels, -np.log(predictions))) / labels.shape[0]


# 如果一个分布的累积分布大于一个随机数r，则返回它的累积下标i
# 这个函数的实际作用是从一个1*27的概率分布中随机抽样一个字母，并返回这个字母的id
def sample_distribution(distribution):
  """Sample one element from a distribution assumed to be an array of normalized probabilities.
  """
  r = random.uniform(0, 1)  #均匀分布
  s = 0
  for i in range(len(distribution)):
    s += distribution[i]
    if s >= r:
      return i
  return len(distribution) - 1

# 把一个预测分布转换为一个1-hot表示，用到了上面这个sample_distribution函数
# 实际作用是从一个1*27的概率分布中随机抽样一个字母，并返回这个字母的1-hot编码
def sample(prediction):
  """Turn a (column) prediction into 1-hot encoded samples."""
  p = np.zeros(shape=[1, vocabulary_size], dtype=np.float)
  p[0, sample_distribution(prediction[0])] = 1.0
  return p

# 生成一个1*27的随机分布，概率和为1
def random_distribution():
  """Generate a random column of probabilities."""
  b = np.random.uniform(0.0, 1.0, size=[1, vocabulary_size])
  return b/np.sum(b, 1)[:,None]


# Simple LSTM Model.  简单的LSTM模型，直接手写LSTM底层代码

# In[9]:

#隐层状态的节点数
num_nodes = 64

# 构建graph
lstm_graph = tf.Graph()
with lstm_graph.as_default():
  # Parameters:这些参数应该是共享的，在训练过程中不断更新。

  # Input gate: input, previous output, and bias.
  # 输入门：输入，前一个输出，偏置
  # ix = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1))
  # im = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
  # ib = tf.Variable(tf.zeros([1, num_nodes]))
    
  # 遗忘门：输入，前一个输出，偏置
  # fx = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1))
  # fm = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
  # fb = tf.Variable(tf.zeros([1, num_nodes]))
  
  # 记忆门：输入，状态，偏置
  # cx = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1))
  # cm = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
  # cb = tf.Variable(tf.zeros([1, num_nodes]))

  # 输出门：输入，前一个输出，偏置
  # ox = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1))
  # om = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
  # ob = tf.Variable(tf.zeros([1, num_nodes]))
    
  x_matrix=tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes*4], -0.1, 0.1))
  m_matrix=tf.Variable(tf.truncated_normal([num_nodes, num_nodes*4], -0.1, 0.1))
  bias_matrix=tf.Variable(tf.zeros([1, num_nodes*4]))

  # 在展开之间保存hidden_value和state_value状态的变量。不可更新
  saved_output = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
  saved_state = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)

  # 分类器权重和偏置
  # Classifier weights and biases.
  w = tf.Variable(tf.truncated_normal([num_nodes, vocabulary_size], -0.1, 0.1))
  b = tf.Variable(tf.zeros([vocabulary_size]))
  

  # Definition of the cell computation.
  # LSTM单元计算的定义。这里的i就是xt，o就是上一次的输出hidden_value，state是上一次的细胞状态
  # 细胞的输出是本次的hidden_value和state
    
  def lstm_cell(i, output_last_time, state_last_time):
    """Create a LSTM cell. See e.g.: http://arxiv.org/pdf/1402.1128v1.pdf
    Note that in this formulation, we omit the various connections between the
    previous state and the gates.
    创建一个LSTM单元，在这里的计算形式当中，忽略了前一个状态和门之间的连接。"""
    # 输入门，遗忘门，更新值，输出门
    # forget_gate =tf.sigmoid(tf.matmul(i, fx) + tf.matmul(output_last_time, fm) + fb)
    # input_gate  =tf.sigmoid(tf.matmul(i, ix) + tf.matmul(output_last_time, im) + ib)
    # update =                tf.matmul(i, cx) + tf.matmul(output_last_time, cm) + cb
    # output_gate =tf.sigmoid(tf.matmul(i, ox) + tf.matmul(output_last_time, om) + ob)
    
    #用两个大矩阵相乘替换原来的四个矩阵乘法：x_matrix   m_matrix
    temp_i_x_result=tf.matmul(i, x_matrix)
    temp_o_x_result=tf.matmul(output_last_time, m_matrix)
    sum_result=temp_i_x_result+temp_o_x_result+bias_matrix
    
    forget_gate= tf.sigmoid( sum_result[:,:num_nodes] )
    input_gate = tf.sigmoid( sum_result[:,num_nodes:num_nodes*2] )
    update =                 sum_result[:,num_nodes*2:num_nodes*3]
    output_gate= tf.sigmoid(sum_result[:,num_nodes*3:] )
    
    # 本细胞内的状态更新，来自两部分
    state = forget_gate * state_last_time + input_gate * tf.tanh(update)
    output=output_gate * tf.tanh(state)
    return output, state

  # Input data.构造输入数据，11*64*27
  train_data = list()
  for _ in range(num_unrollings + 1):
    train_data.append(tf.placeholder(tf.float32, shape=[batch_size,vocabulary_size]))
    
  #构造inputs和labels，差一个时间单位
  train_inputs = train_data[:num_unrollings]
  train_labels = train_data[1:]  
  # labels are inputs shifted by one time step.

  # Unrolled LSTM loop.
  outputs = list()
  output = saved_output
  state = saved_state

  for i in train_inputs: #每一个i是一个64*27矩阵
    output, state = lstm_cell(i, output, state)
    outputs.append(output)

  # State saving across unrollings.
  # 这里看不懂？什么意思？
  with tf.control_dependencies([saved_output.assign(output),saved_state.assign(state)]):
    # Classifier.
    logits = tf.nn.xw_plus_b(tf.concat(0, outputs), w, b)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf.concat(0, train_labels)))

  # Optimizer.优化器，也看不太懂？
  global_step = tf.Variable(0)
  learning_rate = tf.train.exponential_decay(10.0, global_step, 5000, 0.1, staircase=True)
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  gradients, v = zip(*optimizer.compute_gradients(loss))
  gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
  optimizer = optimizer.apply_gradients(zip(gradients, v), global_step=global_step)

  # 下面是预测过程
  # logits是什么形状？
  # 训练集上的预测值
  train_prediction = tf.nn.softmax(logits)
  
  # Sampling and validation eval: batch 1, no unrolling.
  # 抽样，以及验证集评估
  sample_input = tf.placeholder(tf.float32, shape=[1, vocabulary_size])
  saved_sample_output = tf.Variable(tf.zeros([1, num_nodes]))
  saved_sample_state = tf.Variable(tf.zeros([1, num_nodes]))
    
  reset_sample_state = tf.group(
    saved_sample_output.assign(tf.zeros([1, num_nodes])),
    saved_sample_state.assign(tf.zeros([1, num_nodes])))

  sample_output, sample_state = lstm_cell(sample_input, saved_sample_output, saved_sample_state)
    
  with tf.control_dependencies([saved_sample_output.assign(sample_output),saved_sample_state.assign(sample_state)]):
    sample_prediction = tf.nn.softmax(tf.nn.xw_plus_b(sample_output, w, b))


# In[10]:

#迭代步数 报告间隔
num_steps = 10001
summary_frequency = 1000

with tf.Session(graph=lstm_graph) as session:
  # 初始化所有变量
  tf.initialize_all_variables().run()
  print('Initialized')

  # 损失
  mean_loss = 0

  for step in range(num_steps):
    
    # 每个step跑一个batches
    batches = train_batches.next() #11*64*27
    
    feed_dict = dict()
    for i in range(num_unrollings + 1):
      feed_dict[train_data[i]] = batches[i]
    
    # 训练，累加损失l
    _, l, predictions, lr = session.run([optimizer, loss, train_prediction, learning_rate], feed_dict=feed_dict)
    mean_loss += l
    
    #是否报告训练结果
    if step % summary_frequency == 0:
      if step > 0:
        mean_loss = mean_loss / summary_frequency
      # The mean loss is an estimate of the loss over the last few batches.
      print('Average loss at step %d: %f learning rate: %f' % (step, mean_loss, lr))
      mean_loss = 0 #重置maen_loss

      #计算并输出一个mini_batch的perplexity
      labels = np.concatenate(list(batches)[1:])
      print('Minibatch perplexity: %.2f' % float(np.exp(logprob(predictions, labels))))
      
      # 评估验证集的 perplexity.
      reset_sample_state.run()
      valid_logprob = 0
      for _ in range(valid_size): #64
        b = valid_batches.next()
        predictions = sample_prediction.eval({sample_input: b[0]})
        valid_logprob = valid_logprob + logprob(predictions, b[1])
      print('Validation set perplexity: %.2f' % float(np.exp(valid_logprob / valid_size)))
    
      #每10倍间隔就输出一个随机预测结果，这里是1000个step
      
      #这里是怎么做的呢？
      if step % (summary_frequency) == 0:
        # Generate some samples.
        print('=' * 80)
        for _ in range(5):
          feed = sample(random_distribution())
          sentence = characters(feed)[0]
          reset_sample_state.run()
          for _ in range(79):
            prediction = sample_prediction.eval({sample_input: feed})
            feed = sample(prediction)
            sentence += characters(feed)[0]
          print(sentence)
        print('=' * 80)
        
      


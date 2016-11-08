
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


#此版本的bigram_lstm用当前bigram预测后面的bigram,还可以有其他的方案：
#比如ruby, 用ru来预测b,用ub来预测y。这种方案可能更合理！！
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

# filename = maybe_download('/users1/zyli/data/text8.zip', 31344016)
filename = maybe_download('/Users/lzy/data/text8.zip', 31344016)


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

vocabulary_size = (len(string.ascii_lowercase) + 1)*(len(string.ascii_lowercase) + 1) # [a-z] + ' '
# 词表大小共729*729

idx2bi = {}
bi2idx = {}
idx = 0
for i in ' ' + string.ascii_lowercase:
    for j in ' ' + string.ascii_lowercase:
        idx2bi[idx] = i + j
        bi2idx[i + j] = idx
        idx += 1
def bi2id(char):
    if char in bi2idx.keys():
        return bi2idx[char]
    else:
        print('Unexpected character: %s' % char)
        return 0
def id2bi(dictid):
    if 0 <= dictid < len(idx2bi):
        return idx2bi[dictid]
    else:
        return '  '


print(bi2id('ad'), bi2id('zf'), bi2id('  '), bi2id('r '), bi2id('ï'))
print(id2bi(31), id2bi(708), id2bi(0), id2bi(486))

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
    # 每个batch是一个64*729的矩阵，每行表示一个字母
    batch = np.zeros(shape=(self._batch_size), dtype=np.int32)
    
    for b in range(self._batch_size):
      batch[b] = bi2id(self._text[self._cursor[b]:self._cursor[b]+2])
      #更新当前的cursor位置
      self._cursor[b] = (self._cursor[b] + 2) % self._text_size
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
    
    # batches中包含11个64的一维列表
    return batches


# probabilities是一个矩阵，axis=1是往y轴上投影，所以矩阵每一行（729维）表示一个概率分布，
# 返回的列表长度是矩阵的行数，也就是64个字母
def characters(probabilities):
  """Turn a 1-hot encoding or a probability distribution over the possible
  characters back into its (most likely) character representation."""
  return [id2bi(c) for c in probabilities]

# 将好几个batch恢复到它的string形式
def batches2string(batches):
  """Convert a sequence of batches back into their (most likely) string representation."""
  # s先是一个batch的长度，也就是64
  s = [''] * batches[0].shape[0]
  # batches中共有11个64的一维列表
  for b in batches:
    # b是一个batch的概率分布矩阵64
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

#到这里是没有问题的
# In[7]:

#这个函数干什么用的？
def logprob(predictions, labels):
  """Log-probability of the true labels in a predicted batch."""
  # 把predictions里面小于10的-10次方的数值变为10的-10次方
  predictions[predictions < 1e-10] = 1e-10
  return np.sum(np.multiply(labels, -np.log(predictions))) / len(labels)


# 如果一个分布的累积分布大于一个随机数r，则返回它的累积下标i
# 这个函数的实际作用是从一个1*729的概率分布中随机抽样一个字母，并返回这个字母的id
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
# 实际作用是从一个1*729的概率分布中随机抽样一个字母，并返回这个字母的1-hot编码
def sample(prediction):
  """Turn a (column) prediction into 1-hot encoded samples."""
  p = np.zeros(shape=[1, vocabulary_size], dtype=np.float)
  p[0, sample_distribution(prediction[0])] = 1.0
  return p

# 生成一个1*729的随机分布，概率和为1
def random_distribution():
  """Generate a random column of probabilities."""
  b = np.random.uniform(0.0, 1.0, size=[1, vocabulary_size])
  return b/np.sum(b, 1)[:,None]


def one_hot_represent(num,train_labels):
  temp = np.zeros(shape=(num,vocabulary_size), dtype=np.float32)
  return_data=[]
  for m in train_labels:
    for i,j in enumerate(m):
      temp[i,j]=1.0
    return_data.append(temp)
    temp = np.zeros(shape=(num,vocabulary_size), dtype=np.float32)
  return return_data

# Simple LSTM Model.  简单的LSTM模型，直接手写LSTM底层代码

# In[9]:

#模型定义
#隐层状态的节点数
num_nodes = 100
embedding_size=128
# 构建graph
lstm_graph = tf.Graph()
with lstm_graph.as_default():
  x_matrix=tf.Variable(tf.truncated_normal([embedding_size, num_nodes*4], -0.1, 0.1))
  m_matrix=tf.Variable(tf.truncated_normal([num_nodes, num_nodes*4], -0.1, 0.1))
  bias_matrix=tf.Variable(tf.zeros([1, num_nodes*4]))
  saved_output = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
  saved_state = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
  w = tf.Variable(tf.truncated_normal([num_nodes, vocabulary_size], -0.1, 0.1))
  b = tf.Variable(tf.zeros([vocabulary_size]))
    
  def lstm_cell(i, output_last_time, state_last_time, drop):
    if drop:
        i = tf.nn.dropout(i, 0.9)
    # print(i.get_shape(),x_matrix.get_shape())
    temp1=tf.matmul(i, x_matrix)
    temp2=tf.matmul(output_last_time, m_matrix)
    sum_result=temp1+temp2+bias_matrix
    # sum_result=tf.matmul(i, x_matrix)+tf.matmul(output_last_time, m_matrix)+bias_matrix
    forget_gate= tf.sigmoid( sum_result[:,:num_nodes] )
    input_gate = tf.sigmoid( sum_result[:,num_nodes:num_nodes*2] )
    update =                 sum_result[:,num_nodes*2:num_nodes*3]
    output_gate= tf.sigmoid(sum_result[:,num_nodes*3:] )
    # 本细胞内的状态更新，来自两部分
    state = forget_gate * state_last_time + input_gate * tf.tanh(update)
    output= output_gate * tf.tanh(state)
    if drop:
      output = tf.nn.dropout(output, 0.9)  
    return output, state



  # 从这里开始是需要好好修改的

  one_batch = list()
  for _ in range(num_unrollings + 1):
    one_batch.append(tf.placeholder(tf.int32, shape=[batch_size]))
    
  #构造inputs和labels，差一个时间单位
  train_data = one_batch[:num_unrollings]
  train_labels = tf.placeholder(tf.float32, shape=[num_unrollings,batch_size,vocabulary_size])
  embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0)) 
  
  # Unrolled LSTM loop.
  outputs = list()
  output = saved_output
  state = saved_state

  for i in train_data: #每一个i是一个64的list
    # print(i.get_shape(),embeddings.get_shape())
    embed = tf.nn.embedding_lookup(embeddings, i)
    output, state = lstm_cell(embed, output, state,True)
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
  #修改到这里结束
  


  
  # 下面是预测过程
  # logits是什么形状？
  # 训练集上的预测值
  train_prediction = tf.nn.softmax(logits)
  # Sampling and validation eval: batch 1, no unrolling.
  # 抽样，以及验证集评估
  sample_input = tf.placeholder(tf.int32, shape=[1])
  sample_embed=tf.nn.embedding_lookup(embeddings, sample_input)

  saved_sample_output = tf.Variable(tf.zeros([1, num_nodes]),trainable=False)
  saved_sample_state = tf.Variable(tf.zeros([1, num_nodes]),trainable=False)
  reset_sample_state = tf.group(
    saved_sample_output.assign(tf.zeros([1, num_nodes])),
    saved_sample_state.assign(tf.zeros([1, num_nodes])))
  
  sample_output, sample_state = lstm_cell(sample_embed, saved_sample_output, saved_sample_state,False)
  with tf.control_dependencies([saved_sample_output.assign(sample_output),saved_sample_state.assign(sample_state)]):
    sample_prediction = tf.nn.softmax(tf.nn.xw_plus_b(sample_output, w, b))
  




num_steps = 10001
summary_frequency = 100

with tf.Session(graph=lstm_graph) as session:
  tf.initialize_all_variables().run()
  print('Initialized')
  mean_loss = 0
  for step in range(num_steps):
    batches = train_batches.next() #11*64
    feed_dict = dict()
    for i in range(num_unrollings + 1):
      feed_dict[one_batch[i]] = batches[i]
    train_labels_one_not = one_hot_represent(batch_size,batches[1:])
    feed_dict[train_labels]=train_labels_one_not
    # _, l, lr = session.run([optimizer, loss, learning_rate], feed_dict=feed_dict)
    _, l, predictions, lr = session.run([optimizer, loss, train_prediction, learning_rate], feed_dict=feed_dict)
    mean_loss += l

    
    if step % summary_frequency == 0:
      if step > 0:
        mean_loss = mean_loss / summary_frequency
      # The mean loss is an estimate of the loss over the last few batches.
      print('Average loss at step %d: %f learning rate: %f' % (step, mean_loss, lr))
      mean_loss = 0 #重置maen_loss
      
      #计算并输出一个mini_batch的perplexity
      labels = np.concatenate(train_labels_one_not)
      print('Minibatch perplexity: %.2f' % float(np.exp(logprob(predictions, labels))))
      
      # 评估验证集的 perplexity.
      reset_sample_state.run()
      valid_logprob = 0
      for _ in range(valid_size): #64
        b = valid_batches.next()
        predictions = sample_prediction.eval({sample_input: b[0]})
        valid_logprob = valid_logprob + logprob(predictions, one_hot_represent(1,[b[1]]))
      print('Validation set perplexity: %.2f' % float(np.exp(valid_logprob / valid_size)))

      #从这里开始是不需要考虑的
      if step % (summary_frequency) == 0:
        # Generate some samples.
        print('=' * 80)
        for _ in range(5):
          feed = sample(random_distribution())
          sentence = characters([np.argmax(feed)])[0]
          reset_sample_state.run()
          for _ in range(40):
            prediction = sample_prediction.eval({sample_input: [np.argmax(feed)]})
            feed = sample(prediction)
            sentence += characters([np.argmax(feed)])[0]
          print(sentence)
        print('=' * 80)
    
      



from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# In[3]:


import numpy as np
import nltk
import re
import os
import tarfile
import math
import random
import collections


from six.moves import urllib


# In[4]:


import matplotlib as mp
import matplotlib.pyplot as plt
import tensorflow as tf 


# In[5]:


DOWNLOADED_FILENAME = 'ImdbReviews.tar.gz'

def download_file(url_path):
    if not os.path.exists(DOWNLOADED_FILENAME):
        filename, _ = urllib.request.urlretrieve(url_path,DOWNLOADED_FILENAME)
        
    print('found and verified file from this path :', url_path)
    print('Downloaded file', DOWNLOADED_FILENAME)


# In[6]:


TOKEN_REGEX = re.compile("[^A-Za-z0-9 ]+")

def get_reviews(dirname, positive=True):
    label = 1 if positive else 0
    
    reviews = []
    labels = []
    
    for filename in os.listdir(dirname):
        if filename.endswith(".txt"):
            
            with open(dirname + filename , 'rb') as f:
                review = f.read().decode('utf-8')
                review = review.lower().replace("<br />", " ")
                review = re.sub(TOKEN_REGEX, '', review)
                
                #returns tapel of review text formated and a label to say
                #whether it's a postive or negative comment
                reviews.append(review)
                labels.append(label)
                
    return reviews, labels


def extract_reviews():
    
    if not os.path.exists('aclImdb'):
        with tarfile.open(DOWNLOADED_FILENAME) as tar:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar)
            tar.close()
            
    
    positive_reviews = get_reviews("aclImdb/train/pos/", positive=True)
    negative_reviews = get_reviews("aclImdb/train/neg/", positive=False)
    
    return positive_reviews, negative_reviews


# In[7]:


def extract_labels_data():
    
    if not os.path.exists('aclImdb'):
        with tarfile.open(DOWNLOADED_FILENAME) as tar:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar)
            tar.close()
            
    
    positive_reviews, positive_labels = get_reviews("aclImdb/train/pos/", positive=True)
    negative_reviews, negative_labels = get_reviews("aclImdb/train/neg/", positive=False)
    
    data = positive_reviews + negative_reviews
    labels = positive_labels + negative_labels
    
    return  labels, data


# In[8]:


URL_PATH = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
download_file(URL_PATH)


# In[9]:


labels,data = extract_labels_data()


# In[10]:


labels[:10]


# In[ ]:


data[:5]


# In[11]:


max_document_length = max([len(x.split(" ")) for x in data ])
print(max_document_length)


# In[12]:


MAX_SEQUENCE_LENGTH = 250


# In[13]:


vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(MAX_SEQUENCE_LENGTH)


# In[14]:


x_data = np.array(list(vocab_processor.fit_transform(data)))


# In[15]:


y_output = np.array(labels)


# In[16]:


vocabulary_size = len(vocab_processor.vocabulary_)
print(vocabulary_size)


# In[ ]:


x_data[:1]


# In[17]:


np.random.seed(22)
shuffle_indices = np.random.permutation(np.arange(len(x_data)))

x_shuffled = x_data[shuffle_indices]
y_shuffled = y_output[shuffle_indices]


# In[18]:


TRAIN_DATA = 5000
TOTAL_DATA = 6000


train_data = x_shuffled[:TRAIN_DATA]
train_target = y_shuffled[:TRAIN_DATA]

test_data = x_shuffled[TRAIN_DATA:TOTAL_DATA]
test_target = y_shuffled[TRAIN_DATA:TOTAL_DATA]


# In[19]:


tf.reset_default_graph()
x = tf.placeholder(tf.int32, [None,MAX_SEQUENCE_LENGTH] )
y = tf.placeholder(tf.int32, [None] )


# In[20]:


num_epochs = 20
batch_size = 25
embedding_size = 50
max_label = 2



# In[21]:


embedding_matrix = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size],-1.0, 1.0))
embeddings = tf.nn.embedding_lookup(embedding_matrix,x)


# In[22]:


embedding_matrix


# In[24]:


lstmCell = tf.contrib.rnn.BasicLSTMCell(embedding_size)
lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell,output_keep_prob=0.75)


# In[28]:


_, (encoding, _) = tf.nn.dynamic_rnn(lstmCell,embeddings, dtype=tf.float32)


# In[29]:


logits = tf.layers.dense(encoding, max_label,activation=None)


# In[30]:


cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
loss = tf.reduce_mean(cross_entropy)


# In[31]:


prediction = tf.equal(tf.argmax(logits, 1), tf.cast(y,tf.int64))

accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))


# In[32]:


optimizer = tf.train.AdamOptimizer(0.01)
train_step = optimizer.minimize(loss)


# In[33]:


init = tf.global_variables_initializer()


# In[35]:


with tf.Session() as session:
    init.run()
    
    for epoch in range(num_epochs):
        num_batches = int(len(train_data) // batch_size) + 1
        
        for i in range(num_batches):
            min_ix = i*batch_size
            max_ix = np.min([len(train_data), ((i+1) * batch_size)])
            
            x_train_batch = train_data[min_ix:max_ix]
            y_train_batch = train_target[min_ix:max_ix]
            
            train_dict = {x: x_train_batch,  y:y_train_batch}
            session.run(train_step, feed_dict=train_dict)

            train_loss, train_acc = session.run([loss, accuracy],feed_dict = train_dict)
            
        test_dict = {x:test_data, y:test_target}
        test_loss, test_acc = session.run([loss, accuracy],feed_dict = test_dict)
        
        print("Epoch: {} , Test Losses: {:.2}, Test Acc: {:.5}".format(epoch+1,test_loss, test_acc))
            


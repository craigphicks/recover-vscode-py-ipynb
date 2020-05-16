#!/usr/bin/env python
# coding: utf-8

# In[1]:



import math


# In[2]:


import numpy as np
import matplotlib as mp
#get_ipython().run_line_magic('matplotlib', 'inline')
w=1.
minx = np.exp(-1/w)
miny = np.power(minx,w)*np.log(minx)
print('min x,y = <{},{}>'.format(minx,miny))
start = minx*0.0001
x = [0]+[start * np.power(1.001,i) for i in range(10000)]
with np.errstate(divide='ignore', invalid='ignore'):
    y=[np.log(xx)*np.power(xx,w) for xx in x]
    np.nan_to_num(y, copy=False, nan=0.0)
#mp.pyplot.plot(x,y)
#mp.pyplot.xscale("log")
#mp.pyplot.yscale("log")
print('log(1e-60)={}'.format(np.log(1e-60)))
print('1e-60*log(1e-60)={}'.format(1e-60*np.log(1e-60)))


# In[3]:


#https://machinelearningmastery.com/generate-test-datasets-python-scikit-learn/
#sklearn.datasets.make_circles(n_samples=100, *, shuffle=True, noise=None, random_state=None, factor=0.8)
from sklearn.datasets import make_circles
from matplotlib import pyplot
from pandas import DataFrame
# generate 2d classification dataset
X, y = make_circles(n_samples=100, factor=0.5, noise=0.04)
y_sparse = [ [1-yy,yy] for yy in y]
X_test, y_test = make_circles(n_samples=100, factor=0.5, noise=0.04)
y_test_sparse = [ [1-yy,yy] for yy in y_test ]
# scatter plot, dots colored by class value
df = DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
colors = {0:'red', 1:'blue'}
fig, ax = pyplot.subplots()
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
pyplot.show()



# In[8]:


import tensorflow as tf 
class MyLayer(tf.keras.layers.Layer):

  def __init__(self,units, **kwargs):
    super(MyLayer, self).__init__(kwargs)
    w_init = tf.random_normal_initializer()
    input_dim = 2
    self.w = tf.Variable(initial_value=w_init(shape=(input_dim, units),
                                              dtype='float32'),
                         trainable=True)
    b_init = tf.zeros_initializer()
    self.b = tf.Variable(initial_value=b_init(shape=(units,),
                                              dtype='float32'),
                         trainable=True)

  def call(self, inputs):
    return tf.matmul(inputs, self.w) + self.b

class MyModel(tf.keras.Model):

  def __init__(self):
    super(MyModel, self).__init__(name='my_model')
    #self.num_classes = num_classes
    # Define your layers here.
    #self.myLayer1 = MyLayer(512, activation='relu')
    self.myLayer1 = tf.keras.layers.Dense(512)
    self.dense_2 = tf.keras.layers.Dense(2)

  def call(self, inputs):  # this function is called with operator model.()
    # Define your forward pass here,
    # using layers you previously defined (in `__init__`).
    x = self.myLayer1(inputs)
    return self.dense_2(x)

model = MyModel()

# The compile step specifies the training configuration.
#model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
#              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
#              metrics=['accuracy'])

loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

#@tf.function
def train_step(images, labels):
  with tf.GradientTape() as tape:
    # training=True is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(images, training=True)
    loss = loss_object(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)
  train_accuracy(labels, predictions)

    
#@tf.function
def test_step(images, labels):
  # training=False is only needed if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  predictions = model(images, training=False)

EPOCHS=100
for epoch in range(EPOCHS):
  # Reset the metrics at the start of the next epoch
  train_loss.reset_states()
  train_accuracy.reset_states()
  test_loss.reset_states()
  test_accuracy.reset_states()

  train_step(X, y_sparse)

  test_step(X_test, y_test_sparse)

  template = 'Epoch {}, Loss: {:.3f}, Accuracy: {:.3f}, Test Loss: {:.3f}, Test Accuracy: {:.3f}'
  print(template.format(epoch + 1,
                        train_loss.result(),
                        train_accuracy.result() * 100,
                        test_loss.result(),
                        test_accuracy.result() * 100))
#  t_loss = loss_object(labels, predictions)
#  test_loss(t_loss)
#  test_accuracy(labels, predictions)
    


# In[ ]:





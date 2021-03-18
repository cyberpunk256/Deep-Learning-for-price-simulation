#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os.path

from scipy.stats import norm, t
from IPython.core.display import display

get_ipython().run_line_magic('matplotlib', 'inline')

sns.set()

plt.rcParams['font.family'] = 'Yu Mincho'

WORK_DIR = os.path.join('C:\\', 'sample', 'neural_net_simulation')


# In[ ]:


# getting "Dow Jone Industrial Average"
dji = pd.read_csv(
    os.path.join(WORK_DIR, 'data', 'DJI.csv'), 
    parse_dates=['Date'],
    engine='python'
)


# In[ ]:


dji = dji.assign(log_return=np.log(dji.loc[:, 'Adj Close']).diff())
dji = dji.iloc[1:]


# In[ ]:


sns.distplot(dji.loc[1:, 'ログリターン'], fit=norm)


# In[ ]:


sns.distplot(dji.loc[1:, 'ログリターン'], fit=t)


# In[ ]:


class IndexSimulationNet:
    def __init__(
        self,
        sess,
        batch_size,
        lookback_size,
        num_hidden_neurons,
        num_hidden_layers,
        learning_rate, 
        momentum=0.8,
        training=False
    ):
        with tf.variable_scope('input_layer'):
            self._inputs = tf.placeholder(tf.float32, [batch_size, lookback_size], name='inputs')
            self._labels = tf.placeholder(tf.float32, [batch_size, 1], name='label')

        with tf.variable_scope('hidden_layers'):
            self._inputs_normed = tf.layers.batch_normalization(
                tf.expand_dims(self._inputs, axis=-1),
                momentum=momentum,
                training=training
            )
            self._hidden_layer = tf.squeeze(self._inputs_normed, axis=-1)
            
            for layer in range(num_hidden_layers):
                self._hidden_layer = tf.layers.dense(
                    self._hidden_layer, 
                    num_hidden_neurons,
                    activation=tf.tanh
                )
                
        with tf.variable_scope('distribution'):
            self._scale = tf.nn.softplus(
                tf.layers.dense(
                    self._hidden_layer,
                    1,
                    name='scale'
                )
            ) 
            
            self._loc = tf.layers.dense(
                self._hidden_layer, 1, name='loc')
            self._distr = tf.distributions.StudentT(df=3., loc=self._loc, scale=self._scale)
            
        with tf.variable_scope('outputs'):
            self._sample_size = tf.get_variable(
                'sample_size', 1, dtype=tf.int32, trainable=False
            )
            # sampling from distribution
            self._sample = self._distr.sample(self._sample_size)
        
            # average of log liklihood
            self._log_likelihood = tf.reduce_mean(
                tf.log(
                    self._distr.prob(self._labels) 
                )
            )
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            
            with tf.control_dependencies(update_ops):
                self._training_step = tf.train.AdamOptimizer(
                    learning_rate=learning_rate
                ).minimize(
                    -self._log_likelihood  
                )
        
        if training:
            sess.run(tf.global_variables_initializer())
        
        self._saver = tf.train.Saver()
        
    def train_one_step(self, sess, inputs, labels):
        fetches = [
            self._training_step,
            self._log_likelihood
        ]
        feed_dict = {
            self._inputs: inputs,
            self._labels: labels
        }
        return sess.run(fetches, feed_dict)
    
    def sample(self, sess, inputs, sample_size=(1,)):
        return sess.run(
            fetches=[
                self._sample,
                self._scale,
                self._loc,
            ],
            feed_dict={
                self._inputs: inputs,
                self._sample_size: sample_size
            }
        )
    
    def save(self, sess, path):
        self._saver.save(sess, path)
        print(f'model saved in {path}')
        
    def restore(self, sess, path):
        self._saver.restore(sess, path)
        print(f'model restored from {path}')


# In[ ]:


lookback_size = 32
batch_size = 32
num_hidden_neurons = 128
num_hidden_layers = 3
max_learn_steps = 20000
polling_interval = 500
learning_rate = 0.003


# In[ ]:


likelihoods_over_time = np.empty(max_learn_steps // polling_interval)
tf.reset_default_graph()

with tf.Graph().as_default() and tf.Session() as session:
    index_sim_net = IndexSimulationNet(
        session, batch_size, lookback_size,
        num_hidden_neurons, num_hidden_layers, learning_rate,
        training=True
    )
    
    likelihood_mean_polling_interval = 0.
    
    for learn_step in range(max_learn_steps):
        random_row_indices = np.random.choice(
            dji.shape[0] - lookback_size - 1, size=batch_size, replace=True
        )
        random_rows = [
            dji.iloc[random_row:random_row + lookback_size + 1, :] 
            for random_row in random_row_indices
        ]
        log_returns_for_this_step = np.reshape(
           [row[['ログリターン']].values for row in random_rows],
            (batch_size, lookback_size + 1)
        )
        
        _, log_likelihood = index_sim_net.train_one_step(
            session,
            log_returns_for_this_step[:, :lookback_size],
            log_returns_for_this_step[:, lookback_size:]
        )
        likelihood_mean_polling_interval += log_likelihood
        if learn_step % polling_interval == 0:
            likelihoods_over_time[
                learn_step // polling_interval
            ] = likelihood_mean_polling_interval / polling_interval
            
            likelihood_mean_polling_interval = 0.
            print(f'[{(learn_step * 100) // max_learn_steps:3}%] '
                  f'直近{polling_interval}ステップの平均尤度：'
                  f'{likelihoods_over_time[learn_step // polling_interval]:.5f}')
            
    index_sim_net.save(session, os.path.join(WORK_DIR, 'models', 'index_sim_dji'))
likelihoods_over_time_copy = likelihoods_over_time.copy()


# In[ ]:


fig, ax = plt.subplots(figsize=(12, 5))
sns.lineplot(x=list(range(likelihoods_over_time.shape[0])), y=likelihoods_over_time, ax=ax)


# In[ ]:


num_sim_steps = 100
sim_start_index = np.random.randint(
    0,
    dji.shape[0] - lookback_size - num_sim_steps
)
real_log_returns = dji.iloc[
    sim_start_index:sim_start_index + lookback_size + num_sim_steps, :
][['ログリターン']].values.flatten()
gen_log_returns = real_log_returns.copy()
path_dates = dji.iloc[
    sim_start_index:sim_start_index + lookback_size + num_sim_steps, :
][['Date']].values.flatten()


# In[ ]:


gen_scales = np.empty(num_sim_steps)
gen_locs = np.empty_like(gen_scales)
gen_dfs = np.empty_like(gen_scales)

sim_batch_size = 1

print(f'{pd.to_datetime(path_dates[lookback_size]):%Y-%m-%d}'
      f' ～ {pd.to_datetime(path_dates[~0]):%Y-%m-%d}')
tf.reset_default_graph()
with tf.Graph().as_default() and tf.Session() as sess:
    index_sim_net = IndexSimulationNet(
        session,
        sim_batch_size,
        lookback_size,
        num_hidden_neurons,
        num_hidden_layers,
        learning_rate,
        training=False
    )
    index_sim_net.restore(
        sess,
        os.path.join(WORK_DIR, 'models', 'index_sim_dji')
    )
    
    rolling_log_returns = np.array(
        [real_log_returns[:lookback_size]]
    )
    for sim_step in range(num_sim_steps):
        sample, scale, loc = index_sim_net.sample(
            sess,
            rolling_log_returns,
            [1])
        rolling_log_returns[:, :~0] = rolling_log_returns[:, 1:]
        rolling_log_returns[:, ~0] = sample
        
        gen_log_returns[lookback_size + sim_step] = sample
        gen_scales[sim_step] = scale
        gen_locs[sim_step] = loc
    print(f'{num_sim_steps}')


# In[ ]:


both_log_returns = pd.DataFrame(
    data=np.transpose([gen_log_returns, real_log_returns]),
    columns=["simulation value", 'actual value'],
    index=path_dates
)
asset_values = np.exp(both_log_returns).cumprod().stack().reset_index()
asset_values.columns = ['date', 'type', 'asset value']


# In[ ]:


fig, ax = plt.subplots(figsize=(12, 5))
sns.lineplot(data=asset_values, x='date', y='asset value', hue='type', ax=ax)


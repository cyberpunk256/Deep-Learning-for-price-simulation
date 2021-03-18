#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import os.path

get_ipython().run_line_magic('matplotlib', 'inline')

sns.set()

plt.rcParams['font.family'] = 'Yu Mincho'

WORK_DIR = os.path.join('C:\\', 'sample', 'rnn_simulation')

dji = pd.read_csv(
    os.path.join(
        WORK_DIR, 
        'data',
        'DJI.csv'
    ), 
    parse_dates=['Date'],
    engine='python'
)
dji = dji.assign(log_return=np.log(dji.loc[:, 'Adj Close']).diff())
dji = dji.iloc[1:]


# In[ ]:


batch_size = 8
lookback_size = 64
num_hidden_neurons = 40
num_hidden_layers = 3
learning_rate = 1e-5


# In[ ]:


max_epochs = 100
steps_per_epoch = 50


# In[ ]:


likelihoods_over_time = np.empty(max_epochs)
tf.reset_default_graph()

with tf.Graph().as_default() and tf.Session() as session:
    rec_sim_net = RecurrentSimulationNet(
        session, batch_size, lookback_size,
        num_hidden_neurons, num_hidden_layers,
        learning_rate, training=True
    )
    for epoch in range(max_epochs):
        random_row_indices = np.random.choice(
            dji.shape[0] - lookback_size * (steps_per_epoch+1) - 1,
            size=batch_size,
            replace=True
        )
        random_rows = [
            dji.iloc[
                random_row:random_row + lookback_size*(steps_per_epoch+1),
                :
            ]
            for random_row in random_row_indices
        ]
        log_returns_for_this_epoch = np.reshape(
            [row[['log_return']].values for row in random_rows],
            (batch_size, lookback_size*(steps_per_epoch + 1))
        )
        state = None
        likelihood_mean_epoch = 0.
        
        for training_step in range(steps_per_epoch):
            _, log_likelihood, state = rec_sim_net.train_one_step(
                session,
                log_returns_for_this_epoch[
                    :,
                    lookback_size*training_step:lookback_size*(training_step + 1)
                ],
                log_returns_for_this_epoch[
                    :,
                    lookback_size*training_step + 1:lookback_size*(training_step + 1) + 1
                ],
                state=state
            )
            likelihood_mean_epoch += log_likelihood
        likelihoods_over_time[epoch] = likelihood_mean_epoch / steps_per_epoch
        likelihood_mean_polling_interval = 0.
        print(
            f'[{(100*epoch) // max_epochs:3}%] '
            f'{steps_per_epoch}：'
            f'{likelihoods_over_time[epoch]:.5f}'
        )
    rec_sim_net.save(
        session, os.path.join(
            WORK_DIR,
            'models',
            'recurrent_simulation_net'
        )
    )
likelihoods_over_time_copy = likelihoods_over_time.copy()


# In[ ]:


fig, ax = plt.subplots(figsize=(12, 5))
sns.lineplot(
    x=list(range(likelihoods_over_time.shape[0])),
    y=likelihoods_over_time,
    ax=ax
)


# In[ ]:


sim_batch_size = 1
sim_lookback_size = 1
sim_warmup_size = 64 # times for warm up step
sim_steps = 100 # times for simulation

# prepare arrays to store simulation value
en_returns = np.empty((sim_batch_size, sim_warmup_size + sim_steps))

# prepare the array to store actual return
real_returns = np.empty_like(gen_returns)


# In[ ]:


tf.reset_default_graph()
with tf.Graph().as_default() and tf.Session() as session:
    rec_sim_net = RecurrentSimulationNet(
        session, sim_batch_size,
        sim_lookback_size,
        num_hidden_neurons,
        num_hidden_layers,
        learning_rate,
        training=False
    )
    
    rec_sim_net.restore(
        session, os.path.join(
            WORK_DIR, 'models', 'recurrent_simulation_net'
        )
    )
    
    state = None 
    
    random_row_indices = np.random.choice(
        dji.shape[0] - sim_warmup_size - 1,
        size=sim_batch_size,
        replace=True
    )
    
    # prepare the array to store real return
    real_returns[:, :] = np.reshape(
        [
            dji.iloc[
                random_row:random_row + sim_warmup_size + sim_steps, :
            ][['log_return']].values
            for random_row in random_row_indices
        ],
        real_returns.shape
    )
    
    # prepare the actual data for warm-up
    gen_returns[:, :sim_warmup_size] = np.reshape(
        [
            dji.iloc[
                random_row:random_row + sim_warmup_size, :
            ]["log_return"].values
            for random_row in random_row_indices
        ],
        (sim_batch_size, sim_warmup_size))
    
    for warmup_step in range(sim_warmup_size):
        samples, state = rec_sim_net.sample(
            session,
            np.reshape( 
                [
                    dji.iloc[
                        random_row + warmup_step, :]
                    [["log_return"]].values
                    for random_row in random_row_indices
                ],  
                (sim_batch_size, 1)
            ),
            state=state  
        )
    samples = np.squeeze(samples, axis=-1)

    # simulation
    for sim_step in range(sim_steps):
        samples, state = rec_sim_net.sample(
            session,
            samples,  
            state=state
        )
        samples = np.squeeze(samples, axis=-1)
        gen_returns[:, sim_step + sim_warmup_size] = samples


# In[ ]:


real_returns_df = pd.DataFrame(
    np.transpose(real_returns), columns=['real_returns']
)
real_returns_df = real_returns_df.assign(
    real_pf=np.exp(real_returns_df.loc[:, 'real_returns']).cumprod()
)

gen_returns_df = pd.DataFrame(
    np.transpose(gen_returns), columns=['gen_returns']
)
gen_returns_df = gen_returns_df.assign(
    gen_pf=np.exp(gen_returns_df.loc[:, 'gen_returns']).cumprod()
)

sim_data = real_returns_df.join(gen_returns_df)
sim_data = sim_data.loc[
    :, ['real_pf', 'gen_pf']
].stack().reset_index()
sim_data.columns = ['t', 'type', 'pf_value']


# In[ ]:


fig, ax = plt.subplots(figsize=(12, 5))
sns.lineplot(x='t', y='pf_value', hue='type', data=sim_data, ax=ax)


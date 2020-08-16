import numpy as np
from collections import deque
import tensorflow as tf
"""
This file contains the code for our A2C Network, which makes decisions for the
tracing agent. It takes in the state information produced in the step function of
tracing_env, and outputs probabilities for 4 actions (up, right, down, left)

This code does not need to be modified, and its predictions can be accessed via
the 'act' function
"""

class PolicyEstimator_RNN:
    def __init__(self, state_size, action_size,  scope='RNN_model_policy', sess=None, target=None):
        self.state_size = state_size
        self.action_size = action_size
        self.sess = sess or tf.get_default_session()

        with tf.variable_scope(scope):
            self.local_map = tf.placeholder(shape=[None, 5, 625], dtype=tf.float32, name='local_map')

            self.dense1 = tf.contrib.layers.fully_connected(inputs=self.local_map, num_outputs=100)
            self.dense2 = tf.contrib.layers.fully_connected(inputs=self.dense1, num_outputs=100)

            self.state = tf.placeholder(shape=[None, 5, self.state_size],
                                        dtype=tf.float32, name='state')

            self.dense3 = tf.contrib.layers.fully_connected(inputs=self.state, num_outputs=64)
            #self.dense3_5 = tf.contrib.layers.fully_connected(inputs=self.state, num_outputs=64)
            self.dense4 = tf.contrib.layers.fully_connected(inputs=self.dense3, num_outputs=10)

            if target is not None:
                self.final_state = tf.identity(self.dense4)
            else:
                self.final_state = tf.concat([self.dense4, self.dense2], axis=2)

            self.rnn_cell = tf.contrib.rnn.BasicLSTMCell(110)
            self.initial_state = self.rnn_cell.zero_state(batch_size=1, dtype=tf.float32)
            self.rnn_outputs, state = tf.nn.dynamic_rnn(self.rnn_cell, self.final_state, initial_state=self.initial_state)
            self.rnn_outputs = tf.reshape(self.rnn_outputs, [-1, 5 * 110])

            #self.dense6 = tf.contrib.layers.fully_connected(inputs=self.rnn_outputs, num_outputs=64)

            self.output = tf.contrib.layers.fully_connected(inputs=self.rnn_outputs, num_outputs=4)


            self.action_probs = tf.squeeze(tf.nn.softmax(self.output))
            #End of predict step

            #Start of update step
            self.action = tf.placeholder(tf.int32, name='action')
            self.target = tf.placeholder(tf.float32, name='target')

            self.picked_action_prob = tf.gather(self.action_probs, self.action)
            self.picked_action_prob = tf.cond(self.picked_action_prob < 1e-30, lambda: tf.constant(1e-30), lambda: tf.identity(self.picked_action_prob))

            self.loss = -tf.log(self.picked_action_prob) * self.target

            self.optimizer = tf.train.AdamOptimizer(learning_rate=0.00001)
            self.train_op = self.optimizer.minimize(self.loss)


            self.saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope), max_to_keep=20)

            #if self.args.test:
                #self.saver.restore(sess, self.args.weight_dir + policy_weight)




    def predict(self, states, local_map, sess=None):
        sess = sess or tf.get_default_session() or self.sess
        return sess.run(self.action_probs, {self.state: states, self.local_map: local_map})

    def update(self, states, target, action, local_map, sess=None):
        sess = sess or tf.get_default_session() or self.sess

        feed_dict = {self.state: states, self.action: action, self.target: target, self.local_map: local_map}
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed_dict)


        return loss

    def load_weights(self, name, sess):
        self.saver.restore(sess, name)

    def save_weights(self, name, sess, episode=None):
        self.saver.save(sess, name, global_step=episode)



class ValueEstimator_RNN:
    def __init__(self, state_size, action_size, scope='RNN_model_value', sess=None, target=None):
        self.state_size = state_size
        self.action_size = action_size
        self.sess = sess or tf.get_default_session()
        with tf.variable_scope(scope):
            self.local_map = tf.placeholder(shape=[None, 5, 625], dtype=tf.float32, name='local_map')

            self.dense1 = tf.contrib.layers.fully_connected(inputs=self.local_map, num_outputs=100)
            self.dense2 = tf.contrib.layers.fully_connected(inputs=self.dense1, num_outputs=100)

            self.state = tf.placeholder(shape=[None, 5, self.state_size],
                                        dtype=tf.float32, name='state')
            self.dense3 = tf.contrib.layers.fully_connected(inputs=self.state, num_outputs=64)
            #self.dense3_5 = tf.contrib.layers.fully_connected(inputs=self.state, num_outputs=64)
            self.dense4 = tf.contrib.layers.fully_connected(inputs=self.dense3, num_outputs=10)

            if target is not None:
                self.final_state = tf.identity(self.dense4)
            else:
                self.final_state = tf.concat([self.dense4, self.dense2], axis=2)

            self.rnn_cell = tf.contrib.rnn.BasicLSTMCell(110)
            self.initial_state = self.rnn_cell.zero_state(batch_size=1, dtype=tf.float32)
            self.rnn_outputs, state = tf.nn.dynamic_rnn(self.rnn_cell, self.final_state, initial_state=self.initial_state)
            self.rnn_outputs = tf.reshape(self.rnn_outputs, [-1, 5 * 110])

            #self.dense6 = tf.contrib.layers.fully_connected(inputs=self.rnn_outputs, num_outputs=64)

            self.output = tf.contrib.layers.fully_connected(inputs=self.rnn_outputs, num_outputs=1)

            self.value_estimate = tf.squeeze(self.output)
            #End of predict step

            #Start of update step
            self.target = tf.placeholder(tf.float32, name='target')

            self.loss = tf.squared_difference(self.value_estimate, self.target)

            self.optimizer = tf.train.AdamOptimizer(learning_rate=0.00001)
            self.train_op = self.optimizer.minimize(self.loss)

            self.saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope), max_to_keep=10)
            #if self.args.test:
                #self.saver.restore(sess, self.args.weight_dir + value_weight)

    def predict(self, states, local_maps, sess=None):
        sess = sess or tf.get_default_session() or self.sess
        return sess.run(self.value_estimate, {self.state: states, self.local_map: local_maps})

    def update(self, states, target, local_maps, sess=None):
        sess = sess or tf.get_default_session() or self.sess
        feed_dict = {self.state: states, self.target: target, self.local_map: local_maps}
        _, loss = sess.run([self.train_op, self.loss], feed_dict)

        return loss

    def load_weights(self, name, sess):
        self.saver.restore(sess, name)

    def save_weights(self, name, sess, episode=None):
        self.saver.save(sess, name, global_step=episode)

class A2CAgent:
    def __init__(self, state_size, action_size, scope, session, target=None):
        self.state_size = state_size
        self.policy = PolicyEstimator_RNN(state_size, action_size, scope + '_policy', session, target)
        self.value = ValueEstimator_RNN(state_size, action_size, scope + '_value', session, target)
        self.sess = session
        self.memory = deque(maxlen=2000)
        self.load_weight_dir = 'Weights/'
        self.save_weight_dir = 'Weights_save/'

    def act(self, state, local_map):
        action_probs = self.policy.predict(state, local_map, self.sess)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        return action

    def memorize(self, state, local_map, action, reward, next_state, next_local_map, done):
        self.memory.append((state, local_map, action, reward, next_state, next_local_map, done))

    def update(self, states, local_maps, pol_target, val_target, action):
        self.policy.update(states, pol_target, action, local_maps)
        self.value.update(states, val_target, local_maps)

    def load(self, name, name2):
        self.policy.load_weights(self.load_weight_dir + name, self.sess)
        self.value.load_weights(self.load_weight_dir + name2, self.sess)

    def save(self, name, name2, episode=None):
        self.policy.save_weights(self.save_weight_dir + name, self.sess, episode)
        self.value.save_weights(self.save_weight_dir + name2, self.sess, episode)
import random
import numpy as np
from collections import deque
import tensorflow as tf
import os
"""
This file contains the code for our implementation of the DDRQN architecture. This network is used for
the search agent's predictions. This code does not need to be changed. To access its predictions, use the
'act' function
"""

class DDRQNModel:
    def __init__(self, state_size, action_size, scope='DQN_model', sess=None, target=None):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = 0.001
        self.sess = sess or tf.get_default_session()

        with tf.variable_scope(scope):
            self.local_map = tf.placeholder(shape=[None, 5, 625], dtype=tf.float32, name='local_map')

            self.dense1 = tf.contrib.layers.fully_connected(inputs=self.local_map, num_outputs=100)
            self.dense2 = tf.contrib.layers.fully_connected(inputs=self.dense1, num_outputs=100)

            if target is not None:
                self.state = tf.placeholder(shape=[None, self.state_size],
                                            dtype=tf.float32, name='state')
            else:
                self.state = tf.placeholder(shape=[None, 5, self.state_size],
                                            dtype=tf.float32, name='state')

            self.dense3 = tf.contrib.layers.fully_connected(inputs=self.state, num_outputs=64)
            self.dense4 = tf.contrib.layers.fully_connected(inputs=self.dense3, num_outputs=10)

            if target is not None:
                self.final_state = tf.identity(self.dense4)
                self.output = tf.contrib.layers.fully_connected(inputs=self.final_state, num_outputs=self.action_size, activation_fn=None)
            else:
                self.final_state = tf.concat([self.dense4, self.dense2], axis=2)

                self.rnn_cell = tf.contrib.rnn.BasicLSTMCell(110)
                self.initial_state = self.rnn_cell.zero_state(batch_size=1, dtype=tf.float32)
                self.rnn_outputs, state = tf.nn.dynamic_rnn(self.rnn_cell, self.final_state,
                                                            initial_state=self.initial_state)
                self.rnn_outputs = tf.reshape(self.rnn_outputs, [-1, 5 * 110])

                self.output = tf.contrib.layers.fully_connected(inputs=self.rnn_outputs, num_outputs=self.action_size, activation_fn=None)

            self.target = tf.placeholder(shape=[None, self.action_size], dtype=tf.float32, name='target')


            self.loss = tf.squared_difference(self.output, self.target)

            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op = self.optimizer.minimize(self.loss)

            self.saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope), max_to_keep=10)

            # if self.args.test:
            # self.saver.restore(sess, self.args.weight_dir + policy_weight)


    def predict(self, state, local_maps=None, sess=None):
        sess = sess or tf.get_default_session() or self.sess
        if local_maps is not None:
            action_values = sess.run(self.output, {self.state: state, self.local_map:local_maps})
        else:
            action_values = sess.run(self.output, {self.state: state})

        #print(action_values)

        return action_values

    def fit(self, state, target, local_maps=None, sess=None):
        sess = sess or tf.get_default_session() or self.sess
        if local_maps is not None:
            feed_dict = {self.state: state, self.target: target, self.local_map: local_maps}
        else:
            feed_dict = {self.state: state, self.target: target}

        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss


    def load_weights(self, name, sess):
        self.saver.restore(sess, name)


    def save_weights(self, name, sess, episode=None):
        self.saver.save(sess, name, global_step=episode)


class DDRQNAgent:
    def __init__(self, state_size, action_size, scope, session, target=None):
        self.scope = scope
        self.model = DDRQNModel(state_size, action_size, scope + "_model", session, target)
        self.target_model = DDRQNModel(state_size, action_size, scope + "_target", session, target)
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 0.01  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.sess = session

        self.sess.run(tf.global_variables_initializer())

        self.load_weight_dir = "Weights/"
        self.save_weight_dir = "Weights_full/"
        self.temp_weight_dir = "Weights_temp/"

        self.make_dirs()

        self.update_target_model()






        if target is None:
            self.isTarget = False
        else:
            self.isTarget = True


    def update_target_model(self):
        """
        trainable = tf.trainable_variables()
        for i in range(len(trainable) // 2):
            assign_op = trainable[i+len(trainable)//2].assign(trainable[i])
            self.sess.run(assign_op)
        """
        q_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope+"_model")
        q_target_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope+"_target")
        self.sess.run([v_t.assign(v) for v_t, v in zip(q_target_vars, q_vars)])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        if self.isTarget:
            for state, local_map, action, reward, next_state, next_local_map, done in minibatch:
                target = self.model.predict(state)
                target_next = self.model.predict(next_state)
                target_val = self.target_model.predict(next_state)
                if done:
                    target[0][action] = reward
                else:
                    a = np.argmax(target_next[0])
                    target[0][action] = reward + self.gamma * target_val[0][a]

                self.model.fit(state, target)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay


        else:
            for state, local_map, action, reward, next_state, next_local_map, done in minibatch:
                target = self.model.predict(state, local_map)
                target_next = self.model.predict(next_state, next_local_map)
                target_val = self.target_model.predict(next_state, next_local_map)
                if done:
                    target[0][action] = reward
                else:
                    a = np.argmax(target_next[0])
                    target[0][action] = reward + self.gamma * target_val[0][a]

                self.model.fit(state, target, local_map)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def act(self, state, local_maps=None):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, local_maps)
        return np.argmax(act_values[0])  # returns action

    def memorize(self, state, local_map, action, reward, next_state, next_local_map, done):
        self.memory.append((state, local_map, action, reward, next_state, next_local_map, done))

    def make_dirs(self):
        if not os.path.exists(self.load_weight_dir):
            os.makedirs(self.load_weight_dir)

        if not os.path.exists(self.save_weight_dir):
            os.makedirs(self.save_weight_dir)

        if not os.path.exists(self.temp_weight_dir):
            os.makedirs(self.temp_weight_dir)

    def load(self, name, name2):
        self.model.load_weights(self.load_weight_dir + name, self.sess)
        self.target_model.load_weights(self.load_weight_dir + name2, self.sess)

    def save(self, name, name2, episode=None):
        self.model.save_weights(self.save_weight_dir + name, self.sess, episode)
        self.target_model.save_weights(self.save_weight_dir + name2, self.sess, episode)
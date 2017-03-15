#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q3(d): Grooving with GRUs
"""

from __future__ import absolute_import
from __future__ import division

import argparse
import logging
import sys

import tensorflow as tf
import numpy as np

logger = logging.getLogger("hw3.q3.1")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

class MATCHLSTMCell(tf.nn.rnn_cell.RNNCell):
    """Wrapper around our GRU cell implementation that allows us to play
    nicely with TensorFlow.
    """
    def __init__(self, question_size, state_size): 
        self.question_size = question_size
        self._state_size = state_size

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._state_size

    def __call__(self, question_inputs, curr_context_input, last_intermediate_state, scope=None):
        """Updates the state using the previous @state and @inputs.
	question_inputs  # should be (None, state_size, question_size)
	curr_context_input  should be (None, state_size)
	last_state should be (None, state_size)

        Remember the GRU equations are:

        z_t = sigmoid(x_t U_z + h_{t-1} W_z + b_z)
        r_t = sigmoid(x_t U_r + h_{t-1} W_r + b_r)
        o_t = tanh(x_t U_o + r_t * h_{t-1} W_o + b_o)
        h_t = z_t * h_{t-1} + (1 - z_t) * o_t

        TODO: In the code below, implement an GRU cell using @inputs
        (x_t above) and the state (h_{t-1} above).
            - Define W_r, U_r, b_r, W_z, U_z, b_z and W_o, U_o, b_o to
              be variables of the apporiate shape using the
              `tf.get_variable' functions.
            - Compute z, r, o and @new_state (h_t) defined above
        Tips:
            - Remember to initialize your matrices using the xavier
              initialization as before.
        Args:
            inputs: is the input vector of size [None, self.question_size]
            state: is the previous state vector of size [None, self.state_size]
            scope: is the name of the scope to be used when defining the variables inside.
        Returns:
            a pair of the output vector and the new state vector.
        """
        scope = scope or type(self).__name__

	H_q = question_inputs  # should be (batch_size, state_size, question_size)
	h_p = curr_context_input # should be (batch_size, state_size)
	h_r = last_intermediate_state # should be (batch_size, state_size)

        # It's always a good idea to scope variables in functions lest they
        # be defined elsewhere!
        with tf.variable_scope(scope):
            ### YOUR CODE HERE (~20-30 lines)
            xavier_init = tf.contrib.layers.xavier_initializer()
            zero_init = tf.constant_initializer(0)
    
            W_q = tf.get_variable("W_q",shape=(self.state_size, self.state_size,1), initializer=xavier_init)
            W_p = tf.get_variable("W_p",shape=(self.state_size, self.state_size,1), initializer=xavier_init)
            W_r = tf.get_variable("W_r",shape=(self.state_size, self.state_size,1), initializer=xavier_init)

            b_p = tf.get_variable("b_p",shape=(self.state_size,1,1), initializer=zero_init)
            w_g = tf.get_variable("w_g",shape=(self.state_size,1,1), initializer=zero_init)
		
	    print(W_q)

            b_o = tf.get_variable("b_o",shape=(1,), initializer=zero_init)
            	
	    xx = tf.batch_matmul(H_q, W_q)
            insum = ( tf.matmul(h_p, W_p) + tf.matmul(h_r, W_r) + b_p ) # (batch_size, state_size)
	    # G_i = tf.tanh(tf.matmul(H_q, W_q) + )   # broad cast
            # print(G_i)

	    # a_i = tf.nn.softmax(tf.matmul(G_i, tf.transpose(w_g)) + b_o) # broadcast $ (, question_size)

	    # z_i = tf.concat(0, [h_p, tf.matmul(H_q, tf.transpose(a_i))])


            # new_state = tf.nn.rnn.LSTMCell(z_i, h_r) 

	return xx
        # output = new_state
        # return output, new_state

def test_matchlstm_cell():
    with tf.Graph().as_default():
        with tf.variable_scope("test_matchlstm_cell"):
            q_placeholder = tf.placeholder(tf.float32, shape=(None,3,2))
            p_placeholder = tf.placeholder(tf.float32, shape=(None,2))
            h_placeholder = tf.placeholder(tf.float32, shape=(None,2))

            with tf.variable_scope("matchlstm"):
		tf.get_variable("W_q",initializer=np.array(np.eye(2,2), dtype=np.float32))
		tf.get_variable("W_p",initializer=np.array(np.eye(2,2), dtype=np.float32))
		tf.get_variable("W_r",initializer=np.array(np.eye(2,2), dtype=np.float32))
		tf.get_variable("b_p",initializer=np.array(np.ones(2), dtype=np.float32))
		tf.get_variable("w_g",initializer=np.array(np.ones(2), dtype=np.float32))
		tf.get_variable("b_o",initializer=np.array(np.ones(1), dtype=np.float32))


            tf.get_variable_scope().reuse_variables()
            cell = MATCHLSTMCell(3, 2)
            y_var, ht_var = cell(q_placeholder, p_placeholder, h_placeholder, scope="matchlstm")

            init = tf.global_variables_initializer()
            with tf.Session() as session:
                session.run(init)
                q = np.array([
                    [[0.4, 0.5],[0.4, 0.5],[0.4, 0.5]]
                    [[0.3, -0.1],[0.5, 0.6],[0.5, 0.6]]], dtype=np.float32)
                p = np.array([
                    [0.2, 0.5],
                    [-0.3, -0.3]], dtype=np.float32)
                h = np.array([
                    [ 0.320, 0.555],
                    [-0.006, 0.020]], dtype=np.float32)
                ht = h

                y_, ht_ = session.run([y_var, ht_var], feed_dict={q_placeholder: q, p_placholder : p, h_placeholder: h})
                print("y_ = " + str(y_))
                print("ht_ = " + str(ht_))
                assert np.allclose(y_, ht_), "output and state should be equal."
                assert np.allclose(ht, ht_, atol=1e-2), "new state vector does not seem to be correct."

def do_test(_):
    logger.info("Testing gru_cell")
    test_matchlstm_cell()
    logger.info("Passed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tests the GRU cell implemented as part of Q3 of Homework 3')
    subparsers = parser.add_subparsers()

    command_parser = subparsers.add_parser('test', help='')
    command_parser.set_defaults(func=do_test)

    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        ARGS.func(ARGS)


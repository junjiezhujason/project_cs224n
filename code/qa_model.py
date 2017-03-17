from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
import os
from datetime import datetime


import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
from tensorflow.python.ops.rnn_cell import _linear
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs

from evaluate import exact_match_score, f1_score
from util import Progbar, minibatches

from qa_data import PAD_ID, SOS_ID, UNK_ID

logging.basicConfig(level=logging.INFO)

def get_optimizer(opt):
    if opt == "adam":
        optfn = tf.train.AdamOptimizer
    elif opt == "sgd":
        optfn = tf.train.GradientDescentOptimizer
    else:
        assert (False)
    return optfn


class Mixer(object):
    def __init__(self):
        pass

    def mix(self, question_repr, context_paragraph_repr):
        """
        3. Calculate an attention vector over the context paragraph representation based on the question
        representation, or compare the last hidden state of question to all computed paragraph hidden states
        4. Compute a new vector for each context paragraph position that multiplies context-paragraph
        representation with the attention vector.

        Args:
            question_repr: the last hidden state of encoded question
            context_paragraph_repr: all hidden states of encoded context
        Return:
            new context_paragraph_repr weighted by attention
        """
        logging.debug('='*10 + 'Mixer' + '='*10)
        logging.debug('Context paragraph is %s' % str(context_paragraph_repr))
        logging.debug('Question is %s' % str(question_repr))
        a = tf.nn.softmax(tf.matmul(context_paragraph_repr, tf.expand_dims(question_repr, -1)))
        logging.debug('Attention vector is %s' % str(a))
        new_context_paragraph_repr = context_paragraph_repr * a
        logging.debug('New context paragraph is %s' % str(new_context_paragraph_repr))
        return new_context_paragraph_repr

class Encoder(object):
    def __init__(self, size, vocab_dim):
        self.size = size
        self.vocab_dim = vocab_dim

    def encode(self, inputs, seq_len, encoder_state_input):
        """
        In a generalized encode function, you pass in your inputs, seq_len, and an initial hidden state input into this function.

        :param inputs: Symbolic representations of your input
        :param seq_len: this is to make sure tf.nn.dynamic_rnn doesn't iterate through masked steps
        :param encoder_state_input: (Optional) pass this as initial hidden state
                                    to tf.nn.dynamic_rnn to build conditional representations
        :return: an encoded representation of your input.
                 It can be context-level representation, word-level representation,
                 or both.
        """

        logging.debug('='*10 + 'Encoder' + '='*10)
        # Create forward and backward cells
        cell = tf.nn.rnn_cell.LSTMCell(num_units=self.size, state_is_tuple=True)

        # Split initial state
        if encoder_state_input is not None:
            state_fw = encoder_state_input[0]
            state_bw = encoder_state_input[1]
        else:
            state_fw = None
            state_bw = None

        logging.debug('Inputs is %s' % str(inputs))
        # Note input should be padded all to the same length https://piazza.com/class/iw9g8b9yxp46s8?cid=2190
        # inputs: shape (batch_size, max_length, embedding_size)
        hidden_states, final_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell,
                                                                 cell_bw=cell,
                                                                 inputs=inputs,
                                                                 sequence_length=seq_len,
                                                                 initial_state_fw=state_fw,
                                                                 initial_state_bw=state_bw,
                                                                 dtype=tf.float64)

        # Concatenate two end hidden vectors for the final encoded
        # representation of inputs
        concat_hidden_states = tf.concat(2, hidden_states)
        logging.debug('Shape of concatenated BiRNN hidden states is %s' % str(concat_hidden_states))

        final_fw_m_state = final_state[0][1]
        final_bw_m_state = final_state[1][1]
        logging.debug('Shape of BiRNN foward m final_state is %s' % str(final_bw_m_state))
        concat_final_state = tf.concat(1, [final_fw_m_state, final_bw_m_state])
        logging.debug('Shape of concatenated BiRNN final hiden state is %s' % str(concat_final_state))
        return concat_hidden_states, concat_final_state, final_state


class Decoder(object):
    def __init__(self, flag):
        self.config=flag

    def decode(self, knowledge_rep):
        """
        takes in a knowledge representation
        and output a probability estimation over
        all paragraph tokens on which token should be
        the start of the answer span, and which should be
        the end of the answer span.

        Run a final LSTM that does a 2-class classification of these vectors as O or ANSWER.

        :param knowledge_rep: it is a representation of the paragraph and question,
                              decided by how you choose to implement the encoder
        :return:
        """
        logging.debug('='*10 + 'Decoder' + '='*10)
        logging.debug('Input knowledge_rep is %s' % str(knowledge_rep))

        if self.config.model == 'baseline':
            # as = Wahp + W ahq + ba
            # ae = Wehp + W ehq + be
            q, p = knowledge_rep
            logging.debug('Input knowledge_rep q is %s' % str(q))
            logging.debug('Input knowledge_rep p is %s' % str(p))

            xavier_init = tf.contrib.layers.xavier_initializer()
            zero_init = tf.constant_initializer(0)
            Wp_s = tf.get_variable('Wp_s', shape=(self.config.state_size*2, self.config.max_context_length), initializer=xavier_init, dtype=tf.float64)
            Wp_e = tf.get_variable('Wp_e', shape=(self.config.state_size*2, self.config.max_context_length), initializer=xavier_init, dtype=tf.float64)
            Wq_s = tf.get_variable('Wq_s', shape=(self.config.state_size*2, self.config.max_context_length), initializer=xavier_init, dtype=tf.float64)
            Wq_e = tf.get_variable('Wq_e', shape=(self.config.state_size*2, self.config.max_context_length), initializer=xavier_init, dtype=tf.float64)
            b_s  = tf.get_variable('b_s', shape=(self.config.max_context_length, ), initializer=zero_init, dtype=tf.float64)
            b_e  = tf.get_variable('b_e', shape=(self.config.max_context_length, ), initializer=zero_init, dtype=tf.float64)
            with tf.variable_scope('answer_start'):
                a_s = tf.matmul(p, Wp_s) + tf.matmul(q, Wq_s) + b_s
            with tf.variable_scope('answer_scope'):
                a_e = tf.matmul(p, Wp_e) + tf.matmul(q, Wq_e) + b_e
            return a_s, a_e
"""
        cell = tf.nn.rnn_cell.LSTMCell(num_units=1, state_is_tuple=True)
        hidden_states, final_state = tf.nn.dynamic_rnn(cell=cell,
                                                       inputs=knowledge_rep,
                                                       dtype=tf.float64)
        logging.debug('hidden_states is %s' % str(hidden_states))
        xavier_init = tf.contrib.layers.xavier_initializer()
        zero_init = tf.constant_initializer(0)
        b = tf.get_variable('b', shape=(1, ), initializer=zero_init, dtype=tf.float64)
        preds = tf.reduce_mean(tf.sigmoid(hidden_states + b), 2)
        logging.debug('preds is %s' % str(preds))
        # True = Answer, False = Others
        preds = tf.greater_equal(preds, 0.5)
        logging.debug('preds is %s' % str(preds))

        # TODO: figure out how to get the index
        # Index for start of answer is where first 'A' appears
        # s_idx = preds.index(True)
        def true_index(t):
            return tf.reduce_min(tf.where(tf.equal(t, True)))
        s_idx = tf.map_fn(true_index, preds, dtype=tf.int64)
        logging.debug('s_idx is %s' % str(s_idx))

        # Index for end of answer
        # e_idx = preds[s_idx:].index(False) + s_idx
        e_idx = s_idx
        return s_idx, e_idx
"""

class QASystem(object):
    def __init__(self, encoder, mixer, decoder, *args):
        """
        Initializes your System

        :param encoder: an encoder that you constructed in train.py
        :param decoder: a decoder that you constructed in train.py
        :param args: pass in more arguments as needed
        """

        self.encoder = encoder
        self.mixer = mixer
        self.decoder = decoder
        # ==== set up placeholder tokens ========
        # TMP TO REMOVE START
        self.config = args[0]  # FLAG 
        self.pretrained_embeddings = args[1] # embeddings

        # self.saver = args[2]

        # max_question_length = 66
        # max_context_length = 35
        # embedding_size = 50
        # label_size = 2

        # TMP TO REMOVE END
        self.question_placeholder = tf.placeholder(tf.int64, (None, self.config.max_question_length, self.config.n_features))
        print(self.question_placeholder)
        self.question_length_placeholder = tf.placeholder(tf.int64, (None,))
        self.context_placeholder = tf.placeholder(tf.int64, (None, self.config.max_context_length, self.config.n_features))
        self.context_length_placeholder = tf.placeholder(tf.int64, (None,))

        self.start_labels_placeholder=tf.placeholder(tf.int64,(None,))
        self.end_labels_placeholder=tf.placeholder(tf.int64,(None,))
        self.mask_placeholder = tf.placeholder(tf.float64, (None, self.config.max_context_length))

        # ==== assemble pieces ====
        with tf.variable_scope("qa", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            self.setup_embeddings()
        # self.preds = self.setup_system()
        u_pred_s, u_pred_e= self.setup_system()
        self.preds = (self.exp_mask(u_pred_s), self.exp_mask(u_pred_e)) # mask the start end end predictions
        
        self.loss = self.setup_loss(self.preds)

        # ==== set up training/updating procedure ====
        optfn = get_optimizer(self.config.optimizer)
        self.train_op = optfn(self.config.learning_rate).minimize(self.loss)

        self.saver = tf.train.Saver()

    # TODO: add label etc.
    def create_feed_dict(self, 
                         question_batch, 
                         question_length_batch, 
                         context_batch, 
                         context_length_batch,
                         mask_batch=None,
                         labels_batch=None):
        feed_dict = {}
        feed_dict[self.question_placeholder] = question_batch
        feed_dict[self.question_length_placeholder] = question_length_batch
        feed_dict[self.context_placeholder] = context_batch
        feed_dict[self.context_length_placeholder] = context_length_batch
        if labels_batch is not None:
            # labels_batch = np.transpose(labels_batch)
            feed_dict[self.start_labels_placeholder] = labels_batch[0]
            feed_dict[self.end_labels_placeholder] = labels_batch[1]
        if mask_batch is not None:
            feed_dict[self.mask_placeholder] = mask_batch
        return feed_dict

    def attention_flow_layer(self, h, u, h_mask=None, u_mask=None):
        # Query2Context
        with tf.variable_scope("attn_layer"):
            # create bi-attension layer
            # N = tf.shape(self.context_length_placeholder)[0] # batch size
            # JX = tf.shape(h)[1]  # maximum_context_length
            JQ = self.config.max_question_length
            JX = self.config.max_context_length
            d = self.config.state_size*2

	    xavier_init = tf.contrib.layers.xavier_initializer()
	    zero_init = tf.constant_initializer(0)

	    w_s = tf.get_variable('w_s', shape=(3*d, ), initializer=xavier_init, dtype=tf.float64)


            h_aug = tf.tile(tf.expand_dims(h, 2), [1, 1, JQ, 1]) # [?, JX, JQ, d]
            u_aug = tf.tile(tf.expand_dims(u, 1), [1, JX, 1, 1]) # [?. JX, JQ, d]
	    h_dot_u = tf.multiply(h_aug,  u_aug)                 # [?. JX, JQ, d]

            huhu = tf.concat(3, [h_aug, u_aug, h_dot_u])         # [?. JX, JQ, 3d]
            
            logging.info("h_aug:"+str(h_aug))
            logging.info("u_aug:"+str(u_aug))
            logging.info("h_dot_u:"+str(h_dot_u))
            logging.info("huhu:"+str(huhu))

            S_logits = tf.reshape(tf.matmul(tf.reshape(huhu, [-1, 3*d]), tf.expand_dims(w_s,1)), [-1, JX, JQ])  # S_logit to be [N, JX, JQ]
	    logging.info("S_logits: "+str(S_logits))
	     
	    # u_a = softsel(u_aug, S_logits)	
            a_t = tf.nn.softmax(S_logits, -1) # [N, JX, JQ] softmax on the question dimension
            
            # [N, JX, JQ] * [N, JQ, 2*d]
            u_a = tf.matmul(a_t, u)
	    logging.info("u_a: "+str(u_a)) 

            p0 = tf.concat(2, [h, u_a, h * u_a])
	    logging.info("p0: "+str(p0))

        return p0

    

    def setup_system(self):
	"""
        After your modularized implementation of encoder and decoder
        you should call various functions inside encoder, decoder here
        to assemble your reading comprehension system!
        :return:
        """
        question, context = self.setup_embeddings()

        # STEP1: Run a BiLSTM over the question, concatenate the two end hidden
        # vectors and call that the question representation.
        with tf.variable_scope('q'):
            question_length = self.question_length_placeholder  # TODO: name
            question_paragraph_repr, question_repr, q_state = self.encoder.encode(inputs=question,
                                                                    seq_len=question_length,
                                                                    encoder_state_input=None)

        # STEP2: Run a BiLSTM over the context paragraph, conditioned on the
        # question representation.
        with tf.variable_scope('c'):
            context_length = self.context_length_placeholder  # TODO: name
            context_paragraph_repr, context_repr, c_state = self.encoder.encode(inputs=context,
                                                                  seq_len=context_length,
                                                                  encoder_state_input=None)
        # STEP3: Calculate an attention vector over the context paragraph representation based on the question
        # representation.
        # STEP4: Compute a new vector for each context paragraph position that multiplies context-paragraph
        # representation with the attention vector.
        updated_context_paragraph_repr = self.mixer.mix(question_repr, context_paragraph_repr)

	logging.info("Question_paragraph_repr:"+str(question_paragraph_repr))
	logging.info("Context_paragraph_repr:"+str(context_paragraph_repr))
	
	attn_out = self.attention_flow_layer(context_paragraph_repr,question_paragraph_repr)
	
	s_idx, e_idx = self.simple_decoder(attn_out, self.config.state_size*6, self.config.max_context_length)
        
        # s_idx, e_idx = self.decoder.decode((question_repr, context_repr))
        return s_idx, e_idx

    def simple_decoder(self, H_in, d, con_len):
	xavier_init = tf.contrib.layers.xavier_initializer()
	zero_init = tf.constant_initializer(0)

        with tf.variable_scope("simple_decoder"):
            Wp_s = tf.get_variable('Wp_s', shape=(d, ), initializer=xavier_init, dtype=tf.float64)
            Wp_e = tf.get_variable('Wp_e', shape=(d, ), initializer=xavier_init, dtype=tf.float64)
            b_s  = tf.get_variable('b_s', shape=(), initializer=zero_init, dtype=tf.float64)
            b_e  = tf.get_variable('b_e', shape=(), initializer=zero_init, dtype=tf.float64)

            with tf.variable_scope('answer_start'):
                a_s  = tf.reshape(tf.matmul(tf.reshape(H_in, [-1, d]), tf.expand_dims(Wp_s, 1)), [-1, con_len]) + b_s
            with tf.variable_scope('answer_scope'):
                a_e  = tf.reshape(tf.matmul(tf.reshape(H_in, [-1, d]), tf.expand_dims(Wp_e, 1)), [-1, con_len]) + b_e
        return a_s, a_e

    def naive_decoder(self, H_r):
	print("="*10)
	print("NAIVE DECODER")
        d = self.config.state_size*2
        con_len = self.config.max_context_length
        return self.simple_decoder(H_r, d, con_len)

    def exp_mask(self, val):
        """Give very negative number to unmasked elements in val.
            Same shape as val, where some elements are very small (exponentially zero)
        """
        return tf.add(val, self.mask_placeholder)

    def setup_loss(self, preds):
        """
        Set up your loss computation here
        :return:
        """
        with vs.variable_scope("loss"):
            u_pred_s, u_pred_e = preds

            # pred_s = self.exp_mask(u_pred_s)
            # pred_e = self.exp_mask(u_pred_e)
            pred_s = u_pred_s
            pred_e = u_pred_e
            print("LOSS pred_s: "+str(pred_s))
            print("LOSS pred_e: "+str(pred_e))
            
            loss_e = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred_e, labels=self.end_labels_placeholder)

            loss_s = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred_s, labels=self.start_labels_placeholder)

            loss = loss_s + loss_e
        return tf.reduce_mean(loss)
    
        
    def setup_embeddings(self):
        """
        Loads distributed word representations based on placeholder tokens
        :return:
        """
        with vs.variable_scope("embeddings"):
            embedding_tensor = tf.Variable(self.pretrained_embeddings, trainable=False)
            # embedding_tensor = tf.cast(self.pretrained_embeddings, tf.float64)
            question_embedding_lookup = tf.nn.embedding_lookup(embedding_tensor, self.question_placeholder)
            context_embedding_lookup = tf.nn.embedding_lookup(embedding_tensor, self.context_placeholder)
            question_embeddings = tf.reshape(question_embedding_lookup, [-1, self.config.max_question_length, self.config.embedding_size * self.config.n_features])
            context_embeddings = tf.reshape(context_embedding_lookup, [-1, self.config.max_context_length, self.config.embedding_size * self.config.n_features])
        return question_embeddings, context_embeddings

    def optimize(self, session, train_x, train_y):
        """
        Takes in actual data to optimize your model
        This method is equivalent to a step() function
        :return:
        """
        input_feed = {}

        # fill in this feed_dictionary like:
        # input_feed['train_x'] = train_x

        output_feed = []

        outputs = session.run(output_feed, input_feed)

        return outputs

    def test(self, session, valid_x, valid_y):
        """
        in here you should compute a cost for your validation set
        and tune your hyperparameters according to the validation set performance
        :return:
        """
        input_feed = {}

        # fill in this feed_dictionary like:
        # input_feed['valid_x'] = valid_x

        output_feed = []

        outputs = session.run(output_feed, input_feed)

        return outputs

    def decode(self, session, test_x):
        """
        Returns the probability distribution over different positions in the paragraph
        so that other methods like self.answer() will be able to work properly
        :return:
        """
        input_feed = {}

        # fill in this feed_dictionary like:
        # input_feed['test_x'] = test_x

        output_feed = []

        outputs = session.run(output_feed, input_feed)

        return outputs

    def answer(self, session, test_x):

        yp, yp2 = self.decode(session, test_x)

        a_s = np.argmax(yp, axis=1)
        a_e = np.argmax(yp2, axis=1)

        return (a_s, a_e)

    def validate(self, sess, valid_dataset):
        """
        Iterate through the validation dataset and determine what
        the validation cost is.

        This method calls self.test() which explicitly calculates validation cost.

        How you implement this function is dependent on how you design
        your data iteration function

        :return:
        """
        valid_cost = 0

        for valid_x, valid_y in valid_dataset:
          valid_cost = self.test(sess, valid_x, valid_y)


        return valid_cost

    def evaluate_answer(self, session, dataset, sample=100, log=False):
        """
        Evaluate the model's performance using the harmonic mean of F1 and Exact Match (EM)
        with the set of true answer labels

        This step actually takes quite some time. So we can only sample 100 examples
        from either training or testing set.

        :param session: session should always be centrally managed in train.py
        :param dataset: a representation of our data [data_tokenized, data_raw], in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param sample: how many examples in dataset we look at
        :param log: whether we print to std out stream
        :return:
        """
        f1 = []
        em = []
        n_samples = 0
        input_data = dataset[0]
        raw_context = dataset[1]
        for i, output_res in enumerate(self.output(session, input_data)):
            # print(output_res)
            raw_context_i = raw_context[i][1]
            true_labels, pred_labels = output_res
            true_answer = ' '.join(raw_context_i[true_labels[0]:true_labels[1]+1])

            if pred_labels[0] > pred_labels[1]:
                pred_answer = ''
            else:
                if pred_labels[0] >= len(raw_context):
                    pred_answer = ''
                else:
                    pred_answer = ' '.join(raw_context_i[pred_labels[0]:pred_labels[1]+1])
            # Caculate score from golden & predicted answer strings.
            f1.append(f1_score(pred_answer, true_answer))
            em.append(exact_match_score(pred_answer, true_answer))

            n_samples += 1
            if (n_samples == sample):
                break

        f1 = np.mean(f1)
        em = np.mean(em)

        logging.info("F1: {}, EM: {}, for {} samples".format(f1, em, n_samples))

        return f1, em

    def pad_sequence(self, sentence, max_length):
	"""Ensures a seqeunce is of length @max_length by padding it and truncating the rest of the sequence.
	Args:
	    sentence: list of featurized words
	    max_length: the desired length for all input/output sequences.
	Returns:
	    a new sentence and  mask
	    Each of sentence', mask are of length @max_length.
	"""
	# Use this zero vector when padding sequences.
	zero_vector = [PAD_ID] * self.config.n_features
	pad_len = max_length - len(sentence) 
	mask = [0] * len(sentence)
	if pad_len > 0: 
	    p_sentence = sentence + [zero_vector] * pad_len 
	    mask += [-1e10] * pad_len
	else:
	    p_sentence = sentence[:max_length]
	return p_sentence, mask

    def featurize_window(self, sentence, window_size=1):
        # sentence_ = []
        # from util import window_iterator
        # for window in window_iterator(sentence, window_size, beg=start, end=end):
        #     sentence_.append(sum(window, []))
        sentence_ = [[word] for word in sentence]
        return sentence_

    def preprocess_question_answer(self, examples):
	# pad sequences
	ret = []
	for q_sent, q_len, c_sent, c_len, lab in examples:

            if len(c_sent) > self.config.max_context_length:
                logging.info("Ignoring sample with context length: "+str(len(c_sent)))
                continue

            if len(q_sent) > self.config.max_question_length:
                logging.info("Ignoring sample with question length: "+str(len(q_sent)))
                continue

            # window selection
            # TODO: CHANGE LATER
            q_sent = self.featurize_window(q_sent)
            c_sent = self.featurize_window(c_sent)

            # avoid bidirection rnn from complaining
            q_len = min(q_len, self.config.max_question_length)
            c_len = min(c_len, self.config.max_context_length)
            
            # padding
            p_q_sent, _ = self.pad_sequence(q_sent, self.config.max_question_length)
            p_c_sent, c_mask = self.pad_sequence(c_sent, self.config.max_context_length)
            ret.append([p_q_sent, q_len, p_c_sent, c_len, c_mask, lab[0], lab[1]])	
        return np.array(ret)
    

    def train_on_batch(self, sess, q_batch, q_len_batch, c_batch, c_len_batch, mask_batch, start_labels_batch, end_labels_batch):
        feed = self.create_feed_dict(q_batch, 
                                     q_len_batch, 
                                     c_batch, 
                                     c_len_batch, 
                                     mask_batch = mask_batch,
 				     labels_batch=[start_labels_batch, end_labels_batch])
       
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss

    def predict_on_batch(self, sess, q_batch, q_len_batch, c_batch, c_len_batch, mask_batch):
        """
        Return the predicted start index and end index (index NOT onehot).
        """
        feed = self.create_feed_dict(q_batch,
                                     q_len_batch,
                                     c_batch,
                                     c_len_batch,
                                     mask_batch=mask_batch)
        predictions = sess.run([tf.argmax(self.preds[0], axis=1),
                                tf.argmax(self.preds[1], axis=1)], feed_dict=feed)
        # print(predictions)
        # predictions = np.transpose(predictions)
        # print(predictions)
        return predictions

    def run_epoch(self, sess, train_set, valid_set, train_raw, valid_raw):
        train_examples = self.preprocess_question_answer(train_set)
        n_train_examples = len(train_examples)
        #print(train_examples)
        prog = Progbar(target=1 + int(n_train_examples / self.config.batch_size))

        for i, batch in enumerate(minibatches(train_examples, self.config.batch_size)):
            loss = self.train_on_batch(sess, *batch)
            # prog.update(i + 1, [("train loss", loss)])
            # if self.report: self.report.log_train_loss(loss)
            print("train loss", loss)
        print("")

        #logging.info("Evaluating on training data")
        #token_cm, entity_scores = self.evaluate(sess, train_examples, train_examples_raw)
        #logging.debug("Token-level confusion matrix:\n" + token_cm.as_table())
        #logging.debug("Token-level scores:\n" + token_cm.summary())
        #logging.info("Entity level P/R/F1: %.2f/%.2f/%.2f", *entity_scores)
        if self.config.data_size == "tiny":
            logging.info("*****Evaluating on training data*****")
            train_dataset = [train_examples, train_raw]
            _, _ = self.evaluate_answer(sess, train_dataset)

        logging.info("*****Evaluating on validation data*****")
        valid_examples = self.preprocess_question_answer(valid_set)
        valid_dataset = [valid_examples,valid_raw]
        f1, em = self.evaluate_answer(sess, valid_dataset)

        # token_cm, entity_scores = self.evaluate_answer(sess, dev_set, dev_set_raw)
        # logging.debug("Token-level confusion matrix:\n" + token_cm.as_table())
        # logging.debug("Token-level scores:\n" + token_cm.summary())
        # logging.info("Entity level P/R/F1: %.2f/%.2f/%.2f", *entity_scores)

        return f1, em

    def train(self, session, dataset, train_dir):
        """
        Implement main training loop

        TIPS:
        You should also implement learning rate annealing (look into tf.train.exponential_decay)
        Considering the long time to train, you should save your model per epoch.

        More ambitious appoarch can include implement early stopping, or reload
        previous models if they have higher performance than the current one

        As suggested in the document, you should evaluate your training progress by
        printing out information every fixed number of iterations.

        We recommend you evaluate your model performance on F1 and EM instead of just
        looking at the cost.

        :param session: it should be passed in from train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param train_dir: path to the directory where you should save the model checkpoint
        :return:
        """

        # some free code to print out number of parameters in your model
        # it's always good to check!
        # you will also want to save your model parameters in train_dir
        # so that you can use your trained model to make predictions, or
        # even continue training
        
        results_path = os.path.join(train_dir, "{:%Y%m%d_%H%M%S}".format(datetime.now()))
        model_path = results_path 

        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))

        train_set = dataset['training']
        valid_set = dataset['validation']
        train_raw = dataset['training_raw']
        valid_raw = dataset['validation_raw']

        best_score = 0.
	for epoch in range(self.config.epochs):
	    logging.info("Epoch %d out of %d", epoch + 1, self.config.epochs)
            logging.info("Best score so far: "+str(best_score))
	    score, em = self.run_epoch(session, train_set, valid_set, train_raw, valid_raw)
	    if score > best_score:
		best_score = score
	        print("")
		if self.saver:
		    logging.info("New best score! Saving model in %s", model_path)
                    logging.info("f1: "+str(score)+" em:"+str(em))

                    self.saver.save(session, model_path)
	    print("")
	#     if self.report:
	# 	self.report.log_epoch()
	# 	self.report.save()
	return best_score

    def output(self, sess, inputs):
        """
        Reports the output of the model on examples (uses helper to featurize each example).
        """
        prog = Progbar(target=1 + int(len(inputs) / self.config.batch_size))
        
        true = []
        pred = []
        
        # NOTE shuffle = False means everything will be predicting in order
        for i, batch in enumerate(minibatches(inputs, self.config.batch_size, shuffle=False)):
            # Ignore predict
            batch_input = batch[:-2]
            preds_ = self.predict_on_batch(sess, *batch_input)
            pred += list((np.transpose(preds_)))     # pred for this batch
            true += list(np.transpose(batch[-2:])) # true for this batch
            prog.update(i + 1, [])
            # Return context sentence, gold indexes, predicted indexes
            # ret.append([batch[2], batch[-2:], preds_])

        ret = [(true[i], pred[i]) for i in range(len(true))] 
        # print(ret)
        return ret 

# ===============================================================
# Match-LSTM
# ===============================================================


class QASystemMatchLSTM(QASystem):
    def __init__(self, *args):
        """
        Initializes your System

        :param encoder: an encoder that you constructed in train.py
        :param decoder: a decoder that you constructed in train.py
        :param args: pass in more arguments as needed
        """

        # ==== set up placeholder tokens ========
        # TMP TO REMOVE START
        self.config = args[0]  # FLAG 
        self.pretrained_embeddings = args[1] # embeddings
        

        # TMP TO REMOVE END
        self.question_placeholder = tf.placeholder(tf.int64, (None, self.config.max_question_length, self.config.n_features))
        print(self.question_placeholder)
        self.question_length_placeholder = tf.placeholder(tf.int64, (None,))
        self.context_placeholder = tf.placeholder(tf.int64, (None, self.config.max_context_length, self.config.n_features))
        self.context_length_placeholder = tf.placeholder(tf.int64, (None,))

        self.start_labels_placeholder=tf.placeholder(tf.int64,(None,))
        self.end_labels_placeholder=tf.placeholder(tf.int64,(None,))
        self.mask_placeholder = tf.placeholder(tf.float64, (None, self.config.max_context_length))

        # ==== assemble pieces ====
        with tf.variable_scope("qa", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            self.setup_embeddings()
        u_pred_s, u_pred_e= self.setup_system()
        self.preds = (self.exp_mask(u_pred_s), self.exp_mask(u_pred_e)) # mask the start end end predictions
        self.loss = self.setup_loss(self.preds)
        optfn = get_optimizer(self.config.optimizer)
        self.train_op = optfn(self.config.learning_rate).minimize(self.loss)
        self.saver = tf.train.Saver()

    def setup_LSTM_preprocessing_layer(self):
        """
        In a generalized encode function, you pass in your inputs, seq_len, and an initial hidden state input into this function.

        :param inputs: Symbolic representations of your input
        :param seq_len: this is to make sure tf.nn.dynamic_rnn doesn't iterate through masked steps
        :param encoder_state_input: (Optional) pass this as initial hidden state
                                    to tf.nn.dynamic_rnn to build conditional representations
        :return: an encoded representation of your input.
                 It can be context-level representation, word-level representation,
                 or both.
        """

        question, passage = self.setup_embeddings()

        # LSTM Preprocessing Layer for passage
        q_len = self.question_length_placeholder
        p_len = self.context_length_placeholder
        with tf.variable_scope('p'):
            cell_p = tf.nn.rnn_cell.LSTMCell(num_units=self.config.state_size, state_is_tuple=True)
            H_p, _ = tf.nn.dynamic_rnn(cell=cell_p,
                                       inputs=passage,
                                       sequence_length=p_len,
                                       dtype=tf.float64)

        # LSTM Preprocessing Layer for question
        with tf.variable_scope('q'):
            cell_q = tf.nn.rnn_cell.LSTMCell(num_units=self.config.state_size, state_is_tuple=True)
            H_q, _ = tf.nn.dynamic_rnn(cell=cell_q,
                                       inputs=question,
                                       sequence_length=q_len,
                                       dtype=tf.float64)

        return H_p, H_q

    def setup_match_LSTM_layer(self, H_p, H_q):
        zero_init = tf.constant_initializer(0)
        xavier_init = tf.contrib.layers.xavier_initializer()
        state_size = self.config.state_size
        max_question_length = self.config.max_question_length
      
        with tf.variable_scope('match_LSTM'):
            W_q = tf.get_variable('W_q', shape=(self.config.state_size, self.config.state_size), initializer=xavier_init, dtype=tf.float64)
            W_p = tf.get_variable('W_p', shape=(self.config.state_size, self.config.state_size), initializer=xavier_init, dtype=tf.float64)
            W_r = tf.get_variable('W_r', shape=(self.config.state_size, self.config.state_size), initializer=xavier_init, dtype=tf.float64)
            b_p = tf.get_variable('b_p', shape=(self.config.state_size, ), initializer=zero_init, dtype=tf.float64)
            w = tf.get_variable('w', shape=(self.config.state_size, ), initializer=zero_init, dtype=tf.float64)
            b = tf.get_variable('b', shape=(), initializer=zero_init, dtype=tf.float64)

        # ========================================================
        #  MatchLSTMCell class
        #  =======================================================
        class MatchLSTMCell(tf.nn.rnn_cell.BasicLSTMCell):
            def __call__(self, h_p, state):
                """Long short-term memory cell (LSTM)."""
                # Parameters of gates are concatenated into one multiply for efficiency.
                c, h = state
                logging.debug('W_q is ' + str(W_q))
                logging.debug('H_q is ' + str(H_q))

                # H_q was (?, Q, L), change it to (?xQ, L), so we can multiple
                # to W_q (L, L). Then return to (?, Q, L)
                G_part1 = tf.reshape(tf.matmul(tf.reshape(H_q, [-1, state_size]), W_q), [-1, max_question_length, state_size])
                logging.debug('G_1 is' + str(G_part1))
                
                G_part2 = tf.expand_dims(tf.matmul(h_p, W_p) + tf.matmul(h, W_r) + b_p,1)
                logging.debug('G_2 is' + str(G_part2))

                G = tf.tanh(G_part1 + G_part2)
                logging.debug('G is' + str(G))
                
                # G is (?, Q, L), w is (L, 1), reshape G to (?xQ, L) so can
                # multiple with w to get (?xQ, 1), then reshape to get a(?, Q)
                a = tf.nn.softmax(tf.reshape(tf.matmul(tf.reshape(G, [-1, state_size]), tf.expand_dims(w, 1)), [-1, max_question_length]) + b)
                logging.debug('a is' + str(a))
                
                # h_p is (?, L), a is (?, Q), H_q is (?, Q, L)
                # a reshape to (?, 1, Q), multipe to create (?, 1, L), then
                # reshape to (?, L)
                z_part2 = tf.reshape(tf.matmul(tf.expand_dims(a, 1), H_q), [-1, state_size])
                logging.debug('z_part2 is ' + str(z_part2))

                z = tf.concat_v2([h_p, z_part2], axis=1)
                logging.debug('z is ' + str(z))
                return super(MatchLSTMCell, self).__call__(z, state)
        # ========================================================
        #  end MatchLSTMCell class
        #  =======================================================

        with tf.variable_scope('match_LSTM'):
            print("\n"*3)
            print(H_p)
            cell = MatchLSTMCell(num_units=self.config.state_size, state_is_tuple=True)
            H_r_tuple, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell,
                                                     cell_bw=cell,
                                                     inputs=H_p,
                                                     sequence_length=self.context_length_placeholder,
                                                     dtype=tf.float64)
            logging.debug('H_r_tuple is' + str(H_r_tuple))

        H_r = tf.concat(2, H_r_tuple)
        logging.debug('H_r is' + str(H_r))
        return H_r

    def setup_pointer_layer(self, H_r):
        print("="*10)
        print("Pointer Decoder")
        zero_init = tf.constant_initializer(0)
        xavier_init = tf.contrib.layers.xavier_initializer()
        state_size = self.config.state_size
        max_context_length = self.config.max_context_length
      
        with tf.variable_scope('ansptr_LSTM'):
            V = tf.get_variable('V', shape=(2*self.config.state_size, self.config.state_size), initializer=xavier_init, dtype=tf.float64)
            W_a = tf.get_variable('W_a', shape=(self.config.state_size, self.config.state_size), initializer=xavier_init, dtype=tf.float64)
            b_a = tf.get_variable('b_a', shape=(self.config.state_size, ), initializer=zero_init, dtype=tf.float64)
            v =     tf.get_variable('v', shape=(self.config.state_size, ), initializer=zero_init, dtype=tf.float64)
            c_v = tf.get_variable('c_v', shape=(), initializer=zero_init, dtype=tf.float64)

        # ========================================================
        #  AnswerPointerCell class
        #  =======================================================
        class AnsPtrCell(tf.nn.rnn_cell.BasicLSTMCell):
            def __call__(self, dummy, state): # there are no inputs neeed to the cell like before?
                """Long short-term memory cell (LSTM)."""
                # Parameters of gates are concatenated into one multiply for efficiency.
                c, h = state
                logging.debug('V is ' + str(V))
                logging.debug('H_r is ' + str(H_r))

                # H_r is (?, P, 2*L), change it to (?xP, 2*L), so we can multiply
                # with W_q (2*L, L). Then return to (?, P, L)
                F_part1 = tf.reshape(tf.matmul(tf.reshape(H_r, [-1, 2*state_size]), V), [-1, max_context_length, state_size])
                logging.debug('G_1 is' + str(F_part1))
                
                F_part2 = tf.expand_dims(tf.matmul(h, W_a) + b_a, 1)
                logging.debug('G_2 is' + str(F_part2))

                F = tf.tanh(F_part1 + F_part2)
                logging.debug('G is' + str(F))
                
                # F is (?, P, L), v is (L, 1), reshape G to (?xP, L) so can
                # multiple with w to get (?xP, 1), then reshape to get a(?, P)
                pre_softmax_score = tf.reshape(tf.matmul(tf.reshape(F, [-1, state_size]), tf.expand_dims(v, 1)), [-1, max_context_length]) + c_v
                beta = tf.nn.softmax(pre_softmax_score)
                logging.debug('beta is' + str(beta))
                
                # beta reshape to (?, 1, P), multipy with H_r (?, P, 2*L)  to create (?, 1, 2*L), then # reshape to (?, 2*L)
                Hbeta = tf.reshape(tf.matmul(tf.expand_dims(beta, 1), H_r), [-1, 2*state_size])
                logging.debug('Hbeta is ' + str(Hbeta))

                LSTM_output, LSTM_state = super(AnsPtrCell, self).__call__(Hbeta, state)

                return pre_softmax_score, LSTM_state 

            @property
            def output_size(self):
                return max_context_length

        # ========================================================
        #  end AnswerPointerCell class
        #  =======================================================

        with tf.variable_scope('ansptr_LSTM'):
            cell = AnsPtrCell(num_units=self.config.state_size, state_is_tuple=True)
            seq_len_2 =  2*tf.ones((tf.shape(self.context_length_placeholder)))
            input_dummy = tf.zeros((tf.shape(self.context_length_placeholder)[0],2,1))
            print("+"*10)
            print(seq_len_2)
            print(input_dummy)
            beta_log, _ = tf.nn.dynamic_rnn(cell=cell,
                                       inputs=input_dummy, # NOTE this input shouldn't matter? 
                                       sequence_length=seq_len_2, # only unroll twice
                                       dtype=tf.float64)

            logging.debug('beta_log' + str(beta_log))
            # NOTE we only need beta_0 and beta_1 in the intermediate steps in the cell
        return beta_log 



    def setup_system(self):
        H_p, H_q = self.setup_LSTM_preprocessing_layer()
        print('H_q is ' + str(H_q))
        print('H_p is ' + str(H_p))
        H_r_fw = self.setup_match_LSTM_layer(H_p, H_q)

        if self.config.decoder_type == "naive":
            pred_s, pred_e = self.naive_decoder(H_r_fw)
        else:
            beta_log_scores = self.setup_pointer_layer(H_r_fw)
            pred_s, pred_e = tf.split(1, 2, beta_log_scores)
            logging.debug("pred_s is "+str(pred_s))
            logging.debug("pred_e is "+str(pred_e))
            pred_s = tf.reshape(pred_s, [-1,self.config.max_context_length])
            pred_e = tf.reshape(pred_e, [-1,self.config.max_context_length])

        
        logging.debug("pred_s is "+str(pred_s))
        logging.debug("pred_e is "+str(pred_e))
        return pred_s, pred_e 



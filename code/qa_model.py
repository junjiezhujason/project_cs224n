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

from tensorflow.python.ops.gen_math_ops import _batch_mat_mul


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
                                                                 dtype=tf.float32)

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


class QASystem(object):
    def __init__(self, encoder, *args):
        """
        Initializes your System

        :param encoder: an encoder that you constructed in train.py
        :param decoder: a decoder that you constructed in train.py
        :param args: pass in more arguments as needed
        """

        self.encoder = encoder

        # ==== set up placeholder tokens ========
        # TMP TO REMOVE START
        self.config = args[0]  # FLAG 
        self.pretrained_embeddings = args[1] # embeddings
        self.num_per_epoch = args[2]

        # self.saver = args[2]

        # max_question_length = 66
        # max_context_length = 35
        # embedding_size = 50
        # label_size = 2

        self.do_keep_prob_placeholder = tf.placeholder(tf.float32, name='dropout_keep_prob')

        # TMP TO REMOVE END
        self.question_placeholder = tf.placeholder(tf.int64, (None, self.config.max_question_length, self.config.n_features))
        print(self.question_placeholder)
        self.question_length_placeholder = tf.placeholder(tf.int64, (None,))
        self.context_placeholder = tf.placeholder(tf.int64, (None, self.config.max_context_length, self.config.n_features))
        self.context_length_placeholder = tf.placeholder(tf.int64, (None,))

        self.start_labels_placeholder=tf.placeholder(tf.int64,(None,))
        self.end_labels_placeholder=tf.placeholder(tf.int64,(None,))
        self.mask_placeholder = tf.placeholder(tf.float32, (None, self.config.max_context_length))

        # ==== assemble pieces ====
        with tf.variable_scope("qa", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            self.setup_embeddings()
        # self.preds = self.setup_system()
        u_pred_s, u_pred_e= self.setup_system()
        self.preds = (self.exp_mask(u_pred_s), self.exp_mask(u_pred_e)) # mask the start end end predictions
        self.loss = self.setup_loss(self.preds)

        # ==== set up training/updating procedure ====
        optfn = get_optimizer(self.config.optimizer)

        self.global_step = tf.contrib.framework.get_or_create_global_step()
        num_batches_per_epoch = (self.num_per_epoch / self.config.batch_size)
        self.decay_steps = int(num_batches_per_epoch * self.config.num_epochs_per_decay)

        # Decay the learning rate exponentially based on the number of steps.
        self.lr = tf.train.exponential_decay(self.config.learning_rate,
                                             self.global_step,
                                             self.decay_steps,
                                             self.config.learning_rate_decay_factor,
                                             staircase=True)
        tf.summary.scalar('learning_rate', self.lr)

        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
        self.summary_op = tf.summary.merge(summaries)

        self.train_op = optfn(self.lr).minimize(self.loss, global_step=self.global_step)
        self.saver = tf.train.Saver()

    # TODO: add label etc.
    def create_feed_dict(self, 
                         question_batch, 
                         question_length_batch, 
                         context_batch, 
                         context_length_batch,
                         mask_batch=None,
                         labels_batch=None,
                         is_training=False):
        feed_dict = {}
        
        if is_training:
            dropout_keep_prob = self.dropout_keep_prob
        else:
            dropout_keep_prob = 1.0

        # print("Using dropout_keep_prob of {}".format(dropout_keep_prob))

        feed_dict[self.do_keep_prob_placeholder] = dropout_keep_prob

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

    def attention_flow_layer(self, h, u, d, simple=True):
        # h is context
        # u is question
        # d is the embedding dimension for each of them
        # Query2Context
        P = h
        Q = u
        JP = self.config.max_context_length
        JQ = self.config.max_question_length

        with tf.variable_scope("Attention_Layer"):

            xavier_init = tf.contrib.layers.xavier_initializer()
            zero_init = tf.constant_initializer(0)
            
            if simple:
                W_s = tf.get_variable('Ws_s', shape=(2*d, d), initializer=xavier_init, dtype=tf.float32)
                b_s = tf.get_variable('bs_s', shape=(d, ), initializer=xavier_init, dtype=tf.float32)

                QT = tf.transpose(Q, [0,2,1])
                A = tf.nn.softmax(_batch_mat_mul(h, QT))
                C_P = _batch_mat_mul(A, Q)
                logging.info("C_P:"+str(C_P))

                P_concat = tf.concat(2, [C_P, P]) # [N, JP, 2*d]
                logging.info("P_concat:"+str(P_concat))

                PWs = tf.matmul(tf.reshape(P_concat,[-1, 2*d]), W_s) 
                PWs = tf.reshape(PWs, [-1, JP, d])  #[N, JP, d]
                logging.info("P_times_W:"+str(PWs))

                P_out = PWs + b_s

                embed_dim = d

            else:
                w_s = tf.get_variable('w_s', shape=(3*d, ), initializer=xavier_init, dtype=tf.float32)

                h_aug = tf.tile(tf.expand_dims(h, 2), [1, 1, JQ, 1]) # [?, JP, JQ, d]
                u_aug = tf.tile(tf.expand_dims(u, 1), [1, JP, 1, 1]) # [?. JP, JQ, d]
                h_dot_u = tf.multiply(h_aug,  u_aug)                 # [?. JP, JQ, d]

                huhu = tf.concat(3, [h_aug, u_aug, h_dot_u])         # [?. JP, JQ, 3d]
                
                logging.info("h_aug:"+str(h_aug))
                logging.info("u_aug:"+str(u_aug))
                logging.info("h_dot_u:"+str(h_dot_u))
                logging.info("huhu:"+str(huhu))

                S_logits = tf.reshape(tf.matmul(tf.reshape(huhu, [-1, 3*d]), tf.expand_dims(w_s,1)), [-1, JP, JQ])  # S_logit to be [N, JP, JQ]
                logging.info("S_logits: "+str(S_logits))
                 
                # u_a = softsel(u_aug, S_logits)        
                a_t = tf.nn.softmax(S_logits, -1) # [N, JP, JQ] softmax on the question dimension
                
                # [N, JP, JQ] * [N, JQ, 2*d]
                u_a = tf.matmul(a_t, u)
                logging.info("u_a: "+str(u_a)) 

                P_out = tf.concat(2, [h, u_a, h * u_a])
                embed_dim = 3 * d

        logging.info("P_out: "+str(P_out))
        return P_out, embed_dim
    
    def setup_system(self):
        """
        After your modularized implementation of encoder and decoder
        you should call various functions inside encoder, decoder here
        to assemble your reading comprehension system!
        :return:
        """
        question, context = self.setup_embeddings()
        
        do_prob_ph = self.do_keep_prob_placeholder

        # STEP1: Run a BiLSTM over the question, concatenate the two end hidden
        # vectors and call that the question representation.
        with tf.variable_scope('Question_BiLSTM'):
            question_length = self.question_length_placeholder  # TODO: name
            question_paragraph_repr, question_repr, q_state = self.encoder.encode(inputs=question,
                                                                    seq_len=question_length,
                                                                    encoder_state_input=None,
                                                                    keep_prob=do_prob_ph)

        # STEP2: Run a BiLSTM over the context paragraph, conditioned on the
        # question representation.
        with tf.variable_scope('Context_BiLSTM'): 
            context_length = self.context_length_placeholder  # TODO: name
            context_paragraph_repr, context_repr, c_state = self.encoder.encode(inputs=context,
                                                                  seq_len=context_length,
                                                                  encoder_state_input=None,
                                                                  keep_prob=do_prob_ph)
        # STEP3: Calculate an attention vector over the context paragraph representation based on the question
        # representation.
        # STEP4: Compute a new vector for each context paragraph position that multiplies context-paragraph

        logging.info("Question_paragraph_repr:"+str(question_paragraph_repr))
        logging.info("Context_paragraph_repr:"+str(context_paragraph_repr))
        
        encoded_embed_dim = 2 * self.config.state_size
        
        # attention layer
        attn_out, attn_embed_dim = self.attention_flow_layer(context_paragraph_repr, question_paragraph_repr, encoded_embed_dim)
        logging.info("attn_out:"+str(attn_out))
        # s_idx, e_idx = self.simple_decoder(attn_out, self.config.state_size*6, self.config.max_context_length)
        
        # modeling layer multiple stacked LSTMs
        model_layer_out, model_embed_dim = self.model_layer(attn_out, attn_embed_dim)
        logging.info("model_layer_out:"+str(model_layer_out))

        # decoding layer
        max_context_length = self.config.max_context_length
        s_idx, e_idx = self.lstm_decoder(model_layer_out, model_embed_dim, max_context_length)

        # s_idx, e_idx = self.lstm_decoder(attn_out, attn_embed_dim, max_context_length)
        
        return s_idx, e_idx

    def simple_decoder(self, H_in, d, con_len):
        logging.info("="*10+"Simple Decoder"+"="*10)
        xavier_init = tf.contrib.layers.xavier_initializer()
        zero_init = tf.constant_initializer(0)

        with tf.variable_scope("simple_decoder"):
            Wp_s = tf.get_variable('Wp_s', shape=(d, ), initializer=xavier_init, dtype=tf.float32)
            Wp_e = tf.get_variable('Wp_e', shape=(d, ), initializer=xavier_init, dtype=tf.float32)
            b_s  = tf.get_variable('b_s', shape=(), initializer=zero_init, dtype=tf.float32)
            b_e  = tf.get_variable('b_e', shape=(), initializer=zero_init, dtype=tf.float32)

            with tf.variable_scope('answer_start'):
                a_s  = tf.reshape(tf.matmul(tf.reshape(H_in, [-1, d]), tf.expand_dims(Wp_s, 1)), [-1, con_len]) + b_s
            with tf.variable_scope('answer_end'):
                a_e  = tf.reshape(tf.matmul(tf.reshape(H_in, [-1, d]), tf.expand_dims(Wp_e, 1)), [-1, con_len]) + b_e
        return a_s, a_e

    def simple_linear(self, H_in, d_len, c_len, bias=False):
        xavier_init = tf.contrib.layers.xavier_initializer()
        zero_init = tf.constant_initializer(0)
        with tf.variable_scope("linear"):
            Wp = tf.get_variable('Wp', shape=(d_len, ), initializer=xavier_init, dtype=tf.float32) 
            y = tf.reshape(tf.matmul(tf.reshape(H_in, [-1, d_len]), tf.expand_dims(Wp, 1)), [-1, c_len])
            if bias:
                b_s  = tf.get_variable('b_s', shape=(), initializer=zero_init, dtype=tf.float32) 
                y = y + b_s 
        return y

    def model_layer(self, H_in, d_in):
        logging.info("="*10+"Model Layer"+"="*10)
        # takes an input of (N, context_len, d) 
        cell = tf.nn.rnn_cell.LSTMCell(num_units=d_in, state_is_tuple=True)
        # cell = tf.nn.rnn_cell.MultiRNNCell([cell]*2, state_is_tuple=True)
        hidden_states, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell,
                                                           cell_bw=cell,
                                                           inputs=H_in,
                                                           sequence_length=self.context_length_placeholder,
                                                           dtype=tf.float32)
        print("Deep LSTM output: "+str(hidden_states))

        concat_hidden_states = tf.concat(2, hidden_states)
        return concat_hidden_states, d_in*2

    def lstm_decoder(self, H_in, d, con_len):
        logging.info("="*10+"LSTM Decoder"+"="*10)
        xavier_init = tf.contrib.layers.xavier_initializer()
        zero_init = tf.constant_initializer(0)
        logging.info("H_in:"+str(H_in))
        logging.info("d:"+str(d))
        logging.info("con_len:"+str(con_len))
        
        do_prob_ph = self.do_keep_prob_placeholder

        def lstm_decoder_cell(H_in, d, c_len):
            """Helper function to create a lstm decoder."""
            cell = tf.nn.rnn_cell.LSTMCell(num_units=d, state_is_tuple=True)
            H_out, _ = tf.nn.dynamic_rnn(cell=cell,
                                        inputs=H_in,
                                        sequence_length=self.context_length_placeholder,
                                        dtype=tf.float32)
            H_out = tf.nn.dropout(H_out, do_prob_ph, name="H_out_dropout")
            y = self.simple_linear(H_out, d, c_len) 
            return y 

        with tf.variable_scope("answer_start_decoder"):
            a_s = self.simple_linear(H_in, d, con_len) # lstm_decoder_cell(H_in, d)
        with tf.variable_scope("answer_end_decoder"):
            a_e = lstm_decoder_cell(H_in, d, con_len)
        return a_s, a_e

    def naive_decoder(self, H_r, simple=True):
        print("="*10)
        print("NAIVE DECODER")
        d = self.config.state_size*2
        con_len = self.config.max_context_length
        if simple:
            self.simple_decoder(H_r, d, con_len)
        else: 
            self.lstm_decoder(H_r, d, con_len)
        return 

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

            with vs.variable_scope("start_loss"):
                loss_s = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred_s, labels=self.start_labels_placeholder)

            with vs.variable_scope("end_loss"):
                loss_e = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred_e, labels=self.end_labels_placeholder)

            mean_loss_s = tf.reduce_mean(loss_s)
            mean_loss_e = tf.reduce_mean(loss_e)
            loss = loss_s + loss_e
            mean_loss = tf.reduce_mean(loss)

            tf.summary.scalar("cross_entropy_loss_start", mean_loss_s)
            tf.summary.scalar("cross_entropy_loss_end", mean_loss_e)
            tf.summary.scalar("cross_entropy_loss", mean_loss)

        return mean_loss
    
        
    def setup_embeddings(self):
        """
        Loads distributed word representations based on placeholder tokens
        :return:
        """
        with vs.variable_scope("embeddings"):
            embedding_tensor = tf.Variable(self.pretrained_embeddings, trainable=False)
            # embedding_tensor = tf.cast(self.pretrained_embeddings, tf.float32)
            question_embedding_lookup = tf.nn.embedding_lookup(embedding_tensor, self.question_placeholder)
            context_embedding_lookup = tf.nn.embedding_lookup(embedding_tensor, self.context_placeholder)
            question_embeddings = tf.reshape(question_embedding_lookup, [-1, self.config.max_question_length, self.config.embedding_size * self.config.n_features])
            context_embeddings = tf.reshape(context_embedding_lookup, [-1, self.config.max_context_length, self.config.embedding_size * self.config.n_features])
        return question_embeddings, context_embeddings


    def evaluate_answer(self, session, dataset, sample=100, 
                        return_answer_dict=False, is_training=False):
        """
        Evaluate the model's performance using the harmonic mean of F1 and Exact Match (EM)
        with the set of true answer labels

        This step actually takes quite some time. So we can only sample 100 examples
        from either training or testing set.

        :param session: session should always be centrally managed in train.py
        :param dataset: a representation of our data [data_tokenized, data_raw], in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param sample: how many examples in dataset we look at
        :param return_answer_dict: whether we return predicted answer string as a dict of {uuid: answer } or not
        :return:
        """
        f1 = []
        em = []
        n_samples = 0
        input_data = dataset[0] #
        input_raw = dataset[1]

        if return_answer_dict:
            uuids = dataset[2]
            answers = {}
            
        outputs = self.output(session, input_data, is_training=is_training)

        for i, output_res in enumerate(outputs):
            # print(output_res)
            input_raw_i = input_raw[i][1]
            # print(input_raw_i)
            true_labels, pred_labels = output_res
            true_answer = ' '.join(input_raw_i[true_labels[0]:true_labels[1]+1])

            if pred_labels[0] >= len(input_raw_i):
                pred_answer = '<EXCEEDEND>'
            else:
                if pred_labels[0] > pred_labels[1]:
                    pred_answer = '<REVERSEDSE>'
                else:
                    pred_answer = ' '.join(input_raw_i[pred_labels[0]:pred_labels[1]+1])
            # Caculate score from golden & predicted answer strings.
            f1.append(f1_score(pred_answer, true_answer))
            em.append(exact_match_score(pred_answer, true_answer))

            n_samples += 1
            if (n_samples == sample):
                break

            """
            if self.config.data_size == "tiny":
                input_ques_i = input_raw[i][0]
                raw_ques = ' '.join(input_ques_i)
                if (true_labels[0] != pred_labels[0]) or (true_labels[1] != pred_labels[1]):
                    print("-"*30)
                    print("*** QUESTION: "+raw_ques)
                    print("*** TRUE ANSWER: "+true_answer)
                    print("*** TRUE INDEX:  "+str(true_labels))
                    print("*** PRED ANSWER: "+pred_answer)
                    print("*** PRED INDEX:  "+str(pred_labels))
            """
            if return_answer_dict:
                answers[uuids[i]] = pred_answer

        f1 = np.mean(f1)
        em = np.mean(em)

        print("F1: {}, EM: {}, for {} samples".format(f1, em, n_samples))

        if return_answer_dict:
            return answers
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
        mask = [0.0] * len(sentence)
        if pad_len > 0: 
            p_sentence = sentence + [zero_vector] * pad_len 
            mask += [-1e10] * pad_len
        else:
            p_sentence = sentence[:max_length]
        
        # DOUBLE_CHECKED THAT PADDING IS WORKING
        # print("")
        # print(sentence)
        # print(p_sentence)
        # print(mask)

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

            # TODO HANDLE THIS HERE DOUBLE CHECK WITH YIFEI
            """
            if len(c_sent) > self.config.max_context_length:
                if self.config.preprocess_mode == 'train':
                    logging.info("ERROR: Ignoring sample with context length: "+str(len(c_sent)))
                    continue
                elif self.config.preprocess_mode == 'eval':
                    c_sent = c_sent[:self.config_max_context_length]
                else:
                    raise ValueError('Invalid value "%s" for flag preprocess_mode. Choose from train/eval' % self.config.preprocess_mode)

            if len(q_sent) > self.config.max_question_length:
                if self.config.preprocess_mode == 'train':
                    logging.info("ERROR: Ignoring sample with question length: "+str(len(q_sent)))
                    continue
                elif self.config.preprocess_mode == 'eval':
                    q_sent = q_sent[:self.config_max_question_length]
                else:
                    raise ValueError('Invalid value "%s" for flag preprocess_mode. Choose from train/eval' % self.config.preprocess_mode)
            """

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
                                     labels_batch=[start_labels_batch, end_labels_batch],
                                     is_training=True)
       
        _, loss, global_step = sess.run([self.train_op, self.loss, self.global_step], feed_dict=feed)

        # TODO Maybe add logic to prevent saving every batch (might be slow)
        summary_str = sess.run(self.summary_op, feed_dict=feed)
        self.summary_writer.add_summary(summary_str, global_step)

        return loss

    def predict_on_batch(self, sess, q_batch, q_len_batch, c_batch, c_len_batch, mask_batch,
                         is_training=False):
        """
        Return the predicted start index and end index (index NOT onehot).
        """
        feed = self.create_feed_dict(q_batch,
                                     q_len_batch,
                                     c_batch,
                                     c_len_batch,
                                     mask_batch=mask_batch,
                                     is_training=is_training)
        predictions = sess.run([tf.argmax(self.preds[0], axis=1),
                                tf.argmax(self.preds[1], axis=1)], feed_dict=feed)

        # DEBUG
        # start_probs = (sess.run(self.preds[0], feed_dict=feed))
        # end_probs = (sess.run(self.preds[1], feed_dict=feed))
        # for i in range(len(predictions)):
        #     print("---Predict start:"+str(predictions[0][i]))
        #     print(start_probs[i])
        #     print("---Predict end:"+str(predictions[1][i]))
        #     print(end_probs[i])
        #     print(" ")

        return predictions

    def run_epoch(self, sess, train_set, valid_set, train_raw, valid_raw):
        train_examples = self.preprocess_question_answer(train_set)
        n_train_examples = len(train_examples)
        
        # prog = Progbar(target=1 + int(n_train_examples / self.config.batch_size))

        for i, batch in enumerate(minibatches(train_examples, self.config.batch_size)):
            loss = self.train_on_batch(sess, *batch)
            # prog.update(i + 1, [("train loss {} \n".format(loss))])
            # if self.report: self.report.log_train_loss(loss)
            print("Processed {} batches. Train loss: {}".format(i, loss))
        print("")

        #logging.info("Evaluating on training data")
        #token_cm, entity_scores = self.evaluate(sess, train_examples, train_examples_raw)
        #logging.debug("Token-level confusion matrix:\n" + token_cm.as_table())
        #logging.debug("Token-level scores:\n" + token_cm.summary())
        #logging.info("Entity level P/R/F1: %.2f/%.2f/%.2f", *entity_scores)
        if self.config.data_size == "tiny":
            logging.info("*****Evaluating on training data*****")
            train_dataset = [train_examples, train_raw]
            _, _ = self.evaluate_answer(sess, train_dataset, is_training=False)

        logging.info("*****Evaluating on validation data*****")
        valid_examples = self.preprocess_question_answer(valid_set)
        valid_dataset = [valid_examples,valid_raw]
        f1, em = self.evaluate_answer(sess, valid_dataset, is_training=False)

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
        
        self.dropout_keep_prob = self.config.dropout_keep_prob
        #print("dropout_keep_prob: {}".format(self.dropout_keep_prob))

        self.summary_writer = tf.summary.FileWriter(train_dir, session.graph)

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

                    self.saver.save(session, model_path, global_step=self.global_step)
            print("")
        #     if self.report:
        #       self.report.log_epoch()
        #       self.report.save()
        return best_score

    def output(self, sess, inputs, is_training=False):
        """
        Reports the output of the model on examples (uses helper to featurize each example).
        """
        # prog = Progbar(target=1 + int(len(inputs) / self.config.batch_size))
        
        true = []
        pred = []
        
        # NOTE shuffle = False means everything will be predicting in order
        for i, batch in enumerate(minibatches(inputs, self.config.batch_size, shuffle=False)):
            # Ignore predict
            batch_input = batch[:-2]
            preds_ = self.predict_on_batch(sess, *batch_input, is_training=is_training)
            pred += list((np.transpose(preds_)))     # pred for this batch
            true += list(np.transpose(batch[-2:])) # true for this batch
            # prog.update(i + 1, ["\n"])
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
        self.num_per_epoch = args[2]
        
        
        self.do_keep_prob_placeholder = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # TMP TO REMOVE END
        self.question_placeholder = tf.placeholder(tf.int64, (None, self.config.max_question_length, self.config.n_features))
        print(self.question_placeholder)
        self.question_length_placeholder = tf.placeholder(tf.int64, (None,))
        self.context_placeholder = tf.placeholder(tf.int64, (None, self.config.max_context_length, self.config.n_features))
        self.context_length_placeholder = tf.placeholder(tf.int64, (None,))

        self.start_labels_placeholder=tf.placeholder(tf.int64,(None,))
        self.end_labels_placeholder=tf.placeholder(tf.int64,(None,))
        self.mask_placeholder = tf.placeholder(tf.float32, (None, self.config.max_context_length))

        # ==== assemble pieces ====
        with tf.variable_scope("qa", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            self.setup_embeddings()
        u_pred_s, u_pred_e= self.setup_system()
        self.preds = (self.exp_mask(u_pred_s), self.exp_mask(u_pred_e)) # mask the start end end predictions
        self.loss = self.setup_loss(self.preds)
        optfn = get_optimizer(self.config.optimizer)

        self.global_step = tf.contrib.framework.get_or_create_global_step()
        num_batches_per_epoch = (self.num_per_epoch / self.config.batch_size)
        self.decay_steps = int(num_batches_per_epoch * self.config.num_epochs_per_decay)

        # Decay the learning rate exponentially based on the number of steps.
        self.lr = tf.train.exponential_decay(self.config.learning_rate,
                                             self.global_step,
                                             self.decay_steps,
                                             self.config.learning_rate_decay_factor,
                                             staircase=True)
        tf.summary.scalar('learning_rate', self.lr)

        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
        self.summary_op = tf.summary.merge(summaries)

        self.train_op = optfn(self.lr).minimize(self.loss, global_step=self.global_step)
        self.saver = tf.train.Saver()

    def setup_LSTM_preprocessing_layer(self, question, passage):
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

        # LSTM Preprocessing Layer for passage
        q_len = self.question_length_placeholder
        p_len = self.context_length_placeholder
        with tf.variable_scope('p'):
            cell_p = tf.nn.rnn_cell.LSTMCell(num_units=self.config.state_size, state_is_tuple=True)
            H_p, _ = tf.nn.dynamic_rnn(cell=cell_p,
                                       inputs=passage,
                                       sequence_length=p_len,
                                       dtype=tf.float32)

        # LSTM Preprocessing Layer for question
        with tf.variable_scope('q'):
            cell_q = tf.nn.rnn_cell.LSTMCell(num_units=self.config.state_size, state_is_tuple=True)
            H_q, _ = tf.nn.dynamic_rnn(cell=cell_q,
                                       inputs=question,
                                       sequence_length=q_len,
                                       dtype=tf.float32)

        return H_p, H_q

    def setup_match_LSTM_layer(self, H_p, H_q):
        zero_init = tf.constant_initializer(0)
        xavier_init = tf.contrib.layers.xavier_initializer()
        state_size = self.config.state_size
        max_question_length = self.config.max_question_length
              
        with tf.variable_scope('match_LSTM'):
            W_q = tf.get_variable('W_q', shape=(self.config.state_size, self.config.state_size), initializer=xavier_init, dtype=tf.float32)
            W_p = tf.get_variable('W_p', shape=(self.config.state_size, self.config.state_size), initializer=xavier_init, dtype=tf.float32)
            W_r = tf.get_variable('W_r', shape=(self.config.state_size, self.config.state_size), initializer=xavier_init, dtype=tf.float32)
            b_p = tf.get_variable('b_p', shape=(self.config.state_size, ), initializer=zero_init, dtype=tf.float32)
            w = tf.get_variable('w', shape=(self.config.state_size, ), initializer=zero_init, dtype=tf.float32)
            b = tf.get_variable('b', shape=(), initializer=zero_init, dtype=tf.float32)

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
                z_part2 = tf.reshape(_batch_mat_mul(tf.expand_dims(a, 1), H_q), [-1, state_size])
                logging.debug('z_part2 is ' + str(z_part2))

                z = tf.concat_v2([h_p, z_part2], axis=1)
                logging.debug('z is ' + str(z))
                return super(MatchLSTMCell, self).__call__(z, state)
        # ========================================================
        #  end MatchLSTMCell class
        #  =======================================================

        with tf.variable_scope('match_LSTM'):
            print(H_p)
            cell = MatchLSTMCell(num_units=self.config.state_size, state_is_tuple=True)
            H_r_tuple, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell,
                                                     cell_bw=cell,
                                                     inputs=H_p,
                                                     sequence_length=self.context_length_placeholder,
                                                     dtype=tf.float32)
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
            V = tf.get_variable('V', shape=(2*self.config.state_size, self.config.state_size), initializer=xavier_init, dtype=tf.float32)
            W_a = tf.get_variable('W_a', shape=(self.config.state_size, self.config.state_size), initializer=xavier_init, dtype=tf.float32)
            b_a = tf.get_variable('b_a', shape=(self.config.state_size, ), initializer=zero_init, dtype=tf.float32)
            v =     tf.get_variable('v', shape=(self.config.state_size, ), initializer=zero_init, dtype=tf.float32)
            c_v = tf.get_variable('c_v', shape=(), initializer=zero_init, dtype=tf.float32)

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
                                       dtype=tf.float32)

            logging.debug('beta_log' + str(beta_log))
            # NOTE we only need beta_0 and beta_1 in the intermediate steps in the cell
        return beta_log 



    def setup_system(self):
        do_prob_ph = self.do_keep_prob_placeholder

        question, passage = self.setup_embeddings()
        
        question = tf.nn.dropout(question, do_prob_ph, name="question_dropout")
        passage = tf.nn.dropout(passage, do_prob_ph, name="passage_dropout")
        
        H_p, H_q = self.setup_LSTM_preprocessing_layer(question, passage)
        
        H_p = tf.nn.dropout(H_p, do_prob_ph, name="H_p_dropout")
        H_q = tf.nn.dropout(H_q, do_prob_ph, name="H_q_dropout")
        
        print('H_q is ' + str(H_q))
        print('H_p is ' + str(H_p))
        
        H_r_fw = self.setup_match_LSTM_layer(H_p, H_q)        
        H_r_fw_dim = self.config.state_size * 2
        
        H_r_fw = tf.nn.dropout(H_r_fw, do_prob_ph, name="H_r_fw_dropout")


        if self.config.decoder_type == "pointer":
            beta_log_scores = self.setup_pointer_layer(H_r_fw)
            pred_s, pred_e = tf.split(1, 2, beta_log_scores)
            logging.debug("pred_s is "+str(pred_s))
            logging.debug("pred_e is "+str(pred_e))
            pred_s = tf.reshape(pred_s, [-1,self.config.max_context_length])
            pred_e = tf.reshape(pred_e, [-1,self.config.max_context_length])
        else:
            max_context_length = self.config.max_context_length
            # model_layer_out, model_embed_dim = self.model_layer(H_r_fw, H_r_fw_dim)
            # logging.info("model_layer_out:"+str(model_layer_out))
            # decoding layer
            # pred_s, pred_e= self.lstm_decoder(model_layer_out, model_embed_dim, max_context_length)
            pred_s, pred_e= self.lstm_decoder(H_r_fw, H_r_fw_dim, max_context_length)

        
        logging.debug("pred_s is "+str(pred_s))
        logging.debug("pred_e is "+str(pred_e))
        return pred_s, pred_e 



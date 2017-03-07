from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs

from evaluate import exact_match_score, f1_score
from util import Progbar, minibatches


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

    def encode(self, inputs, masks, encoder_state_input):
        """
        In a generalized encode function, you pass in your inputs,
        masks, and an initial
        hidden state input into this function.

        :param inputs: Symbolic representations of your input
        :param masks: this is to make sure tf.nn.dynamic_rnn doesn't iterate
                      through masked steps
        :param encoder_state_input: (Optional) pass this as initial hidden state
                                    to tf.nn.dynamic_rnn to build conditional representations
        :return: an encoded representation of your input.
                 It can be context-level representation, word-level representation,
                 or both.
        """

        logging.debug('='*10 + 'Encoder' + '='*10)
        # Create forward and backward cells
        cell_fw = tf.nn.rnn_cell.LSTMCell(num_units=self.size, state_is_tuple=True)
        cell_bw = tf.nn.rnn_cell.LSTMCell(num_units=self.size, state_is_tuple=True)

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
        hidden_states, final_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,
                                                                 cell_bw=cell_bw,
                                                                 inputs=inputs,
                                                                 sequence_length=masks,
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


class Decoder(object):
    def __init__(self, output_size):
        self.output_size = output_size

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
        cell = tf.nn.rnn_cell.LSTMCell(num_units=1, state_is_tuple=True)
        hidden_states, final_state = tf.nn.dynamic_rnn(cell=cell,
                                                       inputs=knowledge_rep,
                                                       dtype=tf.float32)
        logging.debug('hidden_states is %s' % str(hidden_states))
        xavier_init = tf.contrib.layers.xavier_initializer()
        zero_init = tf.constant_initializer(0)
        b = tf.get_variable('b', shape=(1, ), initializer=zero_init)
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
        max_question_length = 66
        max_context_length = 35
        embedding_size = 50
        # TMP TO REMOVE END
        self.question_placeholder = tf.placeholder(tf.float32, (None, max_question_length, embedding_size))
        self.question_length_placeholder = tf.placeholder(tf.int32, (None,))
        self.context_placeholder = tf.placeholder(tf.float32, (None, max_context_length, embedding_size))
        self.context_length_placeholder = tf.placeholder(tf.int32, (None,))


        # ==== assemble pieces ====
        with tf.variable_scope("qa", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            self.setup_embeddings()
            self.setup_system()
            self.setup_loss()

        # ==== set up training/updating procedure ====
        pass

    # TODO: add label etc.
    def create_feed_dict(self, question_batch, question_length_batch, context_batch, context_length_batch):
        feed_dict = {}
        feed_dict[self.question_placeholder] = question_batch
        feed_dict[self.question_length_batch] = question_length_batch
        feed_dict[self.context_placeholder] = context_batch
        feed_dict[self.context_length_placeholder] = context_length_batch

    def setup_system(self):
        """
        After your modularized implementation of encoder and decoder
        you should call various functions inside encoder, decoder here
        to assemble your reading comprehension system!
        :return:
        """
        # STEP1: Run a BiLSTM over the question, concatenate the two end hidden
        # vectors and call that the question representation.
        with tf.variable_scope('q'):
            question = self.question_placeholder  # TODO: name
            question_length = self.question_length_placeholder  # TODO: name
            question_paragraph_repr, question_repr, q_state = self.encoder.encode(inputs=question,
                                                                    masks=question_length,
                                                                    encoder_state_input=None)

        # STEP2: Run a BiLSTM over the context paragraph, conditioned on the
        # question representation.
        with tf.variable_scope('c'):
            context = self.context_placeholder  # TODO: name
            context_length = self.context_length_placeholder  # TODO: name
            context_paragraph_repr, context_repr, c_state = self.encoder.encode(inputs=context,
                                                                  masks=context_length,
                                                                  encoder_state_input=q_state)
        # STEP3: Calculate an attention vector over the context paragraph representation based on the question
        # representation.
        # STEP4: Compute a new vector for each context paragraph position that multiplies context-paragraph
        # representation with the attention vector.
        updated_context_paragraph_repr = self.mixer.mix(question_repr, context_paragraph_repr)

        # STEP5: Run a final LSTM that does a 2-class classification of these vectors as O or ANSWER.
        s_idx, e_idx = self.decoder.decode(updated_context_paragraph_repr)


    def setup_loss(self):
        """
        Set up your loss computation here
        :return:
        """
        with vs.variable_scope("loss"):
            pass

    def setup_embeddings(self):
        """
        Loads distributed word representations based on placeholder tokens
        :return:
        """
        with vs.variable_scope("embeddings"):
            pass

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
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param sample: how many examples in dataset we look at
        :param log: whether we print to std out stream
        :return:
        """

        f1 = 0.
        em = 0.

        if log:
            logging.info("F1: {}, EM: {}, for {} samples".format(f1, em, sample))

        return f1, em


    def train_on_batch(self, sess, question_batch, context_batch, labels_batch):
        feed = self.create_feed_dict(question_batch, context_batch, labels_batch=labels_batch)
        print("created feed dict")
        loss = 0.00 # TODO: remove later
        # _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        # return loss
        return loss

    def run_epoch(self, sess, train_set, valid_set):
        train_examples = [train_set["question"], train_set["context"], train_set["label"]]
        n_train_examples = len(train_set["labels"])
        prog = Progbar(target=1 + int(n_train_example / self.config.batch_size))
        for i, batch in enumerate(minibatches(train_examples, self.config.batch_size)):
            loss = self.train_on_batch(sess, *batch)
            # prog.update(i + 1, [("train loss", loss)])
            # if self.report: self.report.log_train_loss(loss)
        print("")

        #logger.info("Evaluating on training data")
        #token_cm, entity_scores = self.evaluate(sess, train_examples, train_examples_raw)
        #logger.debug("Token-level confusion matrix:\n" + token_cm.as_table())
        #logger.debug("Token-level scores:\n" + token_cm.summary())
        #logger.info("Entity level P/R/F1: %.2f/%.2f/%.2f", *entity_scores)

        valid_examples = [valid_set["question"], valid_set["context"], valid_set["label"]]
        # logger.info("Evaluating on development data")
        # token_cm, entity_scores = self.evaluate_answer(sess, dev_set, dev_set_raw)
        # logger.debug("Token-level confusion matrix:\n" + token_cm.as_table())
        # logger.debug("Token-level scores:\n" + token_cm.summary())
        # logger.info("Entity level P/R/F1: %.2f/%.2f/%.2f", *entity_scores)

        f1 = 0.00 # TODO: remove later
        # f1 = entity_scores[-1]
        return f1

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

        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))

        train_set = dataset['training']
        valid_set = dataset['validation']

        best_score = 0.
	for epoch in range(self.config.n_epochs):
	    logger.info("Epoch %d out of %d", epoch + 1, self.config.n_epochs)
	    score = self.run_epoch(sess, train_set, valid_set)
	    if score > best_score:
		best_score = score
		# if saver:
		#     logger.info("New best score! Saving model in %s", self.config.model_output)
		#     saver.save(sess, self.config.model_output)
	    print("")
	#     if self.report:
	# 	self.report.log_epoch()
	# 	self.report.save()
	return best_score


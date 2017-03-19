from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import os
import json
import sys
import random
from os.path import join as pjoin

from tqdm import tqdm
import numpy as np
from six.moves import xrange
import tensorflow as tf

from qa_model import Encoder, QASystem, QASystemMatchLSTM
from preprocessing.squad_preprocess import data_from_json, maybe_download, squad_base_url, \
    invert_map, tokenize, token_idx_map
import qa_data
from data_util import load_glove_embeddings
from evaluate import exact_match_score, f1_score

import logging

logging.basicConfig(level=logging.INFO)

FLAGS = tf.app.flags.FLAGS

# load flags from log file

tf.app.flags.DEFINE_string("config_path", "", "Path to the JSON to load config flags")
tf.app.flags.DEFINE_string("dev_path", "data/squad/dev-v1.1.json", "Path to the JSON dev set to evaluate against (default: ./data/squad/dev-v1.1.json)")
tf.app.flags.DEFINE_string("train_dir", "", "Path to the training directory where the checkpoint is saved")
tf.app.flags.DEFINE_bool("eval_on_train", False, "Run qa_answer to evaluate on train.")
# tf.app.flags.DEFINE_string("embed_path", "", "Path to the trimmed GLoVe embedding (default: ./data/squad/glove.trimmed.{embedding_size}.npz)")

# tf.app.flags.DEFINE_float("learning_rate", 0.003, "Learning rate.")
# tf.app.flags.DEFINE_float("dropout", 0.15, "Fraction of units randomly dropped on non-recurrent connections.")
# tf.app.flags.DEFINE_integer("batch_size", 10, "Batch size to use during training.")
# tf.app.flags.DEFINE_integer("epochs", 0, "Number of epochs to train.")
# tf.app.flags.DEFINE_integer("state_size", 100, "Size of each model layer.")
# tf.app.flags.DEFINE_integer("embedding_size", 100, "Size of the pretrained vocabulary.")
# tf.app.flags.DEFINE_integer("output_size", 750, "The output size of your model.")
# tf.app.flags.DEFINE_integer("keep", 0, "How many checkpoints to keep, 0 indicates keep all.")
# tf.app.flags.DEFINE_string("vocab_path", "data/squad/vocab.dat", "Path to vocab file (default: ./data/squad/vocab.dat)")
# Added
# tf.app.flags.DEFINE_integer("max_question_length", 60, "Maximum question length to consider.")
# tf.app.flags.DEFINE_integer("max_context_length", 300, "Maximum context length to consider.")
# tf.app.flags.DEFINE_integer("label_size", 2, "Dimension of the predicted labels that can be mapped to start-end postion in context.")
# tf.app.flags.DEFINE_integer("n_features", 1, "Number of features to include for each word in the sentence.")
# tf.app.flags.DEFINE_integer("window_length", 1, "Number of features to include for each word in the sentence.")
# tf.app.flags.DEFINE_string("model", "baseline", "Model to use.")
# tf.app.flags.DEFINE_string("optimizer", "adam", "adam / sgd")
# tf.app.flags.DEFINE_string("decoder_type", "pointer", "pointer/naive.")
# tf.app.flags.DEFINE_string("data_size", "tiny", "tiny/full.")
# tf.app.flags.DEFINE_string("preprocess_mode", "eval", "train/eval.")
# tf.app.flags.DEFINE_string("train_dir", "train", "Training directory (default: ./train).")
# tf.app.flags.DEFINE_string("log_dir", "log", "Path to store log and flag files (default: ./log)")

def initialize_model(session, model, train_dir):
    print(train_dir)
    ckpt = tf.train.get_checkpoint_state(train_dir)
    v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
    print(v2_path)
    if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
        logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        assert False, 'Cannot restore chkpt'
        logging.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        logging.info('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))
    return model


def initialize_vocab(vocab_path):
    if tf.gfile.Exists(vocab_path):
        rev_vocab = []
        with tf.gfile.GFile(vocab_path, mode="rb") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip('\n') for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocab_path)


def read_dataset(dataset, tier, vocab):
    """Reads the dataset, extracts context, question, answer,
    and answer pointer in their own file. Returns the number
    of questions and answers processed for the dataset"""
    context_word_cnt = 0
    context_ukn_word_cnt = 0

    context_tokens_data = []
    context_data = []
    question_tokens_data = []
    query_data = []
    question_uuid_data = []
    rand_max = len(vocab.values())
    context_lengths = []

    if FLAGS.eval_on_train:
        s_labels = []
        e_labels = []
        true_answers = []

    for articles_id in tqdm(range(len(dataset['data'])), desc="Preprocessing {}".format(tier)):
        article_paragraphs = dataset['data'][articles_id]['paragraphs']
        for pid in range(len(article_paragraphs)):
            context = article_paragraphs[pid]['context']
            # The following replacements are suggested in the paper
            # BidAF (Seo et al., 2016)
            context = context.replace("''", '" ')
            context = context.replace("``", '" ')

            context_tokens = tokenize(context)

            qas = article_paragraphs[pid]['qas']
            
            for qid in range(len(qas)):
                context_lengths.append(len(context_tokens))
                question = qas[qid]['question']
                question_tokens = tokenize(question)
                question_uuid = qas[qid]['id']
                
                
                context_ids = [vocab.get(w, qa_data.UNK_ID) for w in context_tokens]
                question_ids = [vocab.get(w, qa_data.UNK_ID) for w in question_tokens]
                context_word_cnt += len(context_ids)
                for i in xrange(len(context_ids)):
                    if context_ids[i] == qa_data.UNK_ID:
                        context_ids[i] = random.randint(0, rand_max-1)
                        context_ukn_word_cnt += 1
                        #print(context_tokens[i])
                #print('Percentage of unknow is ' + str(context_ukn_word_cnt/context_word_cnt))

                for i in xrange(len(question_ids)):
                    if int(question_ids[i]) == qa_data.UNK_ID:
                        question_ids[i] = str(random.randint(0, rand_max-1))


                context_data.append(context_ids)
                query_data.append(question_ids)
                question_uuid_data.append(question_uuid)
                context_tokens_data.append(context_tokens)
                question_tokens_data.append(question_tokens)

                if FLAGS.eval_on_train:
                    answer = qas[qid]['answers'][0]['text'].split()
                    # Wrong because qas[qid]['answers'][0]['answer_start'] is the token, not index
                    #s_labels.append(qas[qid]['answers'][0]['answer_start'])
                    #e_labels.append(qas[qid]['answers'][0]['answer_start'] + len(answer) - 1)
                    true_answers.append(answer)
    print(sorted(context_lengths))
    context_lengths_over = [context_length>300 for context_length in context_lengths]
    print(sum(context_lengths_over)/len(context_lengths))

    if FLAGS.eval_on_train:
        return context_tokens_data, context_data, question_tokens_data, query_data, question_uuid_data, s_labels, e_labels, true_answers
    return context_tokens_data, context_data, question_tokens_data, query_data, question_uuid_data



def prepare_dev(prefix, dev_filename, vocab):
    # Don't check file size, since we could be using other datasets
    dev_dataset = maybe_download(squad_base_url, dev_filename, prefix)

    dev_data = data_from_json(os.path.join(prefix, dev_filename))

    if FLAGS.eval_on_train:
        context_tokens_data, context_data, question_tokens_data, question_data, question_uuid_data, s_labels, e_labels, true_answers = read_dataset(dev_data, 'train', vocab)
        return context_tokens_data, context_data, question_tokens_data, question_data, question_uuid_data, s_labels, e_labels, true_answers
    context_tokens_data, context_data, question_tokens_data, question_data, question_uuid_data = read_dataset(dev_data, 'dev', vocab)

    return context_tokens_data, context_data, question_tokens_data, question_data, question_uuid_data


def evaluate_answers(sess, model, dataset):
    answers = {}

    context_tokens_data, context_data, question_tokens_data, question_data, question_uuid_data, s_labels, e_labels, true_answers = dataset
    
    context_data_truncated = []
    context_len_data_truncated = []
    question_data_truncated = []
    question_len_data_truncated = []
    data_size = len(context_data)
    # Split the string of index in context_data nad question_data, and convert them to integer
    # create truncated version of context and question
    for i in range(data_size):
        if len(context_data[i]) > FLAGS.max_context_length:
            context_data_truncated.append(context_data[i][:FLAGS.max_context_length])
            context_len_data_truncated.append(FLAGS.max_context_length)
        else:
            context_data_truncated.append(context_data[i])
            context_len_data_truncated.append(len(context_data[i]))

        if len(question_data[i]) > FLAGS.max_question_length:
            question_data_truncated.append(question_data[i][:FLAGS.max_question_length])
            question_len_data_truncated.append(FLAGS.max_question_length)
        else:
            question_data_truncated.append(question_data[i])
            question_len_data_truncated.append(len(question_data[i]))


    # Pad input data with model.preprocess_question_answer
    data_start = 0
    data_size_to_run = 500
    f1 = []
    em = []
    data_set = [(question_data_truncated[i], 
                 question_len_data_truncated[i], 
                 context_data_truncated[i],
                 context_len_data_truncated[i], 
                 [None, None]) for i in xrange(data_start, data_start + data_size_to_run)] # TODO CHANGE ME
    padded_inputs = model.preprocess_question_answer(data_set) # 7 things per item
    outputs = model.output(sess, padded_inputs)
    for i, output_res in enumerate(outputs):
        true_labels, pred_labels = output_res
        start_idx = pred_labels[0]
        end_idx = pred_labels[1]
        context_len = context_len_data_truncated[i]
        
        if (start_idx >= context_len):
            print('ERROR: start_idx %d exceend context_len %d, this should not happen' % (start_idx, context_len)) 
            answer = '\<EXCEED\>'
        elif (start_idx > end_idx):
            # print(start_idx)
            # print(end_idx)
            answer = '\<REVERSED\>'
        else:
            # TOCHECK how are their golden answer generated?
            # Use rev_vocab to reverse look up vocab from index token
            # answer = ' '.join([rev_vocab[vocab_idx] for vocab_idx in context_data[i][start_idx: end_idx+1]])
            # Use original context
             answer = ' '.join(context_tokens_data[data_start + i][start_idx: end_idx+1])
        

        f1_single = f1_score(answer, ' '.join(true_answers[data_start + i]))
        em_single = exact_match_score(answer, ' '.join(true_answers[data_start + i]))
        if f1_single != 1:
            print('='*100)
            print(' '.join(question_tokens_data[data_start + i]))
            print('Predicted answer is     : %s' % answer)
            print('Original True answer is : %s' % ' '.join(true_answers[data_start + i]))
        f1.append(f1_single)
        em.append(em_single)
        print('f1 score on this data is ' + str(f1_single) + ', em score is ' + str(em_single))


    f1 = np.mean(f1)
    em = np.mean(em)

    logging.info("F1: {}, EM: {}".format(f1, em))



def generate_answers(sess, model, dataset, rev_vocab):
    """
    Loop over the dev or test dataset and generate answer.

    Note: output format must be answers[uuid] = "real answer"
    You must provide a string of words instead of just a list, or start and end index

    In main() function we are dumping onto a JSON file

    evaluate.py will take the output JSON along with the original JSON file
    and output a F1 and EM

    You must implement this function in order to submit to Leaderboard.

    :param sess: active TF session
    :param model: a built QASystem model
    :param rev_vocab: this is a list of vocabulary that maps index to actual words
    :return:
    """
    answers = {}

    context_tokens_data, context_data, question_tokens_data, question_data, question_uuid_data = dataset
    
    context_data_truncated = []
    context_len_data_truncated = []
    question_data_truncated = []
    question_len_data_truncated = []
    data_size = len(context_data)
    # Split the string of index in context_data nad question_data, and convert them to integer
    # create truncated version of context and question
    for i in range(data_size):
        if len(context_data[i]) > FLAGS.max_context_length:
            context_data_truncated.append(context_data[i][:FLAGS.max_context_length])
            context_len_data_truncated.append(FLAGS.max_context_length)
        else:
            context_data_truncated.append(context_data[i])
            context_len_data_truncated.append(len(context_data[i]))

        if len(question_data[i]) > FLAGS.max_question_length:
            question_data_truncated.append(question_data[i][:FLAGS.max_question_length])
            question_len_data_truncated.append(FLAGS.max_question_length)
        else:
            question_data_truncated.append(question_data[i])
            question_len_data_truncated.append(len(question_data[i]))





    # Pad input data with model.preprocess_question_answer
    fake_label = [None, None]
    data_set = [(question_data_truncated[i], 
                 question_len_data_truncated[i], 
                 context_data_truncated[i],
                 context_len_data_truncated[i], 
                 fake_label) for i in xrange(data_size)] # TODO CHANGE ME
    padded_inputs = model.preprocess_question_answer(data_set) # 7 things per item
    outputs = model.output(sess, padded_inputs)
    for i, output_res in enumerate(outputs):
        _, pred_labels = output_res
        start_idx = pred_labels[0]
        end_idx = pred_labels[1]
        context_len = context_len_data_truncated[i]
        
        if (start_idx >= context_len):
            print('ERROR: start_idx %d exceend context_len %d, this should not happen' % (start_idx, context_len)) 
            answer = '\<EXCEED\>'
        elif (start_idx > end_idx):
            # print(start_idx)
            # print(end_idx)
            answer = '\<REVERSED\>'
        else:
            # TOCHECK how are their golden answer generated?
            # Use rev_vocab to reverse look up vocab from index token
            # answer = ' '.join([rev_vocab[vocab_idx] for vocab_idx in context_data[i][start_idx: end_idx+1]])
            # Use original context
            answer = ' '.join(context_tokens_data[i][start_idx: end_idx+1])
        print('Answer is %s' % answer)
        answers[question_uuid_data[i]] = answer
    return answers


def get_normalized_train_dir(train_dir):
    """
    Adds symlink to {train_dir} from /tmp/cs224n-squad-train to canonicalize the
    file paths saved in the checkpoint. This allows the model to be reloaded even
    if the location of the checkpoint files has moved, allowing usage with CodaLab.
    This must be done on both train.py and qa_answer.py in order to work.
    """
    global_train_dir = '/tmp/cs224n-squad-train'
    if os.path.exists(global_train_dir):
        os.unlink(global_train_dir)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    os.symlink(os.path.abspath(train_dir), global_train_dir)
    return global_train_dir


def process_dev_json_to_files():
    # dev_path example data/squad/dev-v1.1.json
    download_prefix = os.path.dirname(os.path.abspath(FLAGS.dev_path)) # data/squad/
    dev_filename = os.path.basename(FLAGS.dev_path) # "dev-v1.1.json"
    # relative path to save the data
    data_prefix = os.path.join("data", "squad", "qa_answer")


    print("Downloading datasets into {}".format(download_prefix))
    print("Preprocessing datasets into {}".format(data_prefix))

    if not os.path.exists(download_prefix):
        os.makedirs(download_prefix)
    if not os.path.exists(data_prefix):
        os.makedirs(data_prefix)

    maybe_download(squad_base_url, dev_filename, download_prefix, None)
    # Read data from dev json file
    dev_data = data_from_json(os.path.join(download_prefix, dev_filename))
    # write data out to data_prefix location
    dev_num_questions, dev_num_answers = read_write_dataset(dev_data, 'dev', data_prefix)

def main(_):

    config_fname = FLAGS.config_path
    assert os.path.exists(config_fname), "config file does not exist"
    logging.info("Loaded configs from: "+config_fname)
    with open(config_fname,"rb") as fp:
        json_flag = json.load(fp)
    # print(json_flag)
    print(vars(FLAGS))
    for key, value in json_flag.iteritems():
        if key=="train_dir":
            continue
        FLAGS.__setattr__(key, value)
        
    print(vars(FLAGS))
    assert os.path.exists(FLAGS.train_dir), "train dir does not exist"
    # assert False
    
    FLAGS.eval_on_train = True

    vocab, rev_vocab = initialize_vocab(FLAGS.vocab_path)
    embed_path = FLAGS.embed_path or pjoin("data", "squad", "glove.trimmed.{}.npz".format(FLAGS.embedding_size))

    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    file_handler = logging.FileHandler(pjoin(FLAGS.log_dir, "log.txt"))
    logging.getLogger().addHandler(file_handler)

    with open(os.path.join(FLAGS.log_dir, "flags.json"), 'w') as fout:
        json.dump(FLAGS.__flags, fout)

    # ========= Load Dataset =========
    # You can change this code to load dataset in your own way
    
    # ============ EVAL_ON_TRAIN: START =============
    if FLAGS.eval_on_train:
        FLAGS.dev_path = "download/squad/dev-v1.1.json"
        print('='*50 + '\nLoad dev-v1.1.json for evalulating model.')

    dev_dirname = os.path.dirname(os.path.abspath(FLAGS.dev_path))
    dev_filename = os.path.basename(FLAGS.dev_path)
    dataset = prepare_dev(dev_dirname, dev_filename, vocab)
    if FLAGS.eval_on_train:
        context_tokens_data, context_data, question_tokens_data, question_data, question_uuid_data, s_labels, e_labels, true_answers = dataset
    else:
        context_tokens_data, context_data, question_tokens_data, question_data, question_uuid_data = dataset


    
    # FLAGS.max_context_length = max(context_len_data) #700 in this dataset.
    # For baseline, because the linear decoder weights dimension is max_length x state_size*2, it need to the max of both dataset
    if FLAGS.max_context_length==0:
        raise ValueError('max_context_length is set to be zero now. Please set it with --max_context_length flag.\n FYI the max length in training set is 766, in dev set is 700.')
    if FLAGS.max_question_length==0:
        raise ValueError('max_question_length is set to be zero now. Please set it with --max_question_length flag.\n')

    for i in range(1):
      logging.debug('context')
      logging.debug(' '.join(context_tokens_data[i]))
      logging.debug('context_data')
      logging.debug(context_data[i])
      logging.debug('question')
      logging.debug(' '.join(question_tokens_data[i]))
      logging.debug('question_data')
      logging.debug(question_data[i])
      logging.debug('uuid_data')
      logging.debug(question_uuid_data[i])

    # ========= Model-specific =========
    # You must change the following code to adjust to your model
    embed_path = FLAGS.embed_path or pjoin("data", "squad", "glove.trimmed.{}.npz".format(FLAGS.embedding_size))
    embeddings = load_glove_embeddings(embed_path)
    encoder = Encoder(size=FLAGS.state_size, vocab_dim=FLAGS.embedding_size)

    # mixer = Mixer()
    # decoder = Decoder(FLAGS)
    if FLAGS.model == 'baseline':
        qa = QASystem(encoder, None, None, FLAGS, embeddings, 1)
    elif FLAGS.model == 'matchLSTM':
        qa = QASystemMatchLSTM(FLAGS, embeddings, 1)


    with tf.Session() as sess:
        # train_dir = get_normalized_train_dir(FLAGS.train_dir)
        train_dir = FLAGS.train_dir 
        initialize_model(sess, qa, train_dir)
        print('About to start generate_answers')
        print(FLAGS.eval_on_train)
        if FLAGS.eval_on_train:
            evaluate_answers(sess, qa, dataset)
            return

        answers = generate_answers(sess, qa, dataset, rev_vocab)

        # write to json file to root dir
        with io.open('dev-prediction.json', 'w', encoding='utf-8') as f:
            f.write(unicode(json.dumps(answers, ensure_ascii=False)))


if __name__ == "__main__":
  tf.app.run()

"""
Run Bayesian optimization to learn to learn select data for transfer learning.

Uses Python 3.5.
"""

import os
import argparse
import logging
import pickle
import copy

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
import robo
# from robo.fmin import bayesian_optimization

import task_utils
import data_utils
import similarity
import features
from constants import FEATURE_SETS, SENTIMENT, POS, POS_BILSTM, PARSING,\
    TASK2TRAIN_EXAMPLES, TASK2DOMAINS, TASKS, POS_PARSING_TRG_DOMAINS,\
    SENTIMENT_TRG_DOMAINS, BASELINES, BAYES_OPT, RANDOM, MOST_SIMILAR_DOMAIN,\
    MOST_SIMILAR_EXAMPLES, ALL_SOURCE_DATA

from bist_parser.bmstparser.src.utils import ConllEntry


def task2_objective_function(task):
    """Returns the objective function of a task."""
    if task == SENTIMENT:
        return objective_function_sentiment
    if task == POS:
        return objective_function_pos
    if task == POS_BILSTM:
        return objective_function_pos_bilstm
    if task == PARSING:
        return objective_function_parsing
    raise ValueError('No objective function implemented for %s.' % task)


def objective_function_sentiment(feature_weights):
    """
    The objective function to optimize for sentiment analysis.
    :param feature_weights: a numpy array; these are the weights of the features
                            that we want to learn
    :return: the error that should be minimized
    """
    train_subset, train_labels_subset = task_utils.get_data_subsets(
        feature_values, feature_weights, X_train, y_train, SENTIMENT,
        TASK2TRAIN_EXAMPLES[SENTIMENT])

    # train and evaluate the SVM; we input the test documents here but only
    # minimize the validation error
    val_accuracy, _ = task_utils.train_and_evaluate_sentiment(
        train_subset, train_labels_subset, X_val, y_val, X_test, y_test)

    # we minimize the error; the lower the better
    error = 1 - float(val_accuracy)
    return error


def objective_function_pos(feature_weights):
    """
    The objective function to optimize for POS tagging.
    :param feature_weights: a numpy array; these are the weights of the features
                            that we want to learn
    :return: the error that should be minimized
    """
    train_subset, train_labels_subset = task_utils.get_data_subsets(
        feature_values, feature_weights, X_train, y_train, POS,
        TASK2TRAIN_EXAMPLES[POS])

    # train and evaluate the tagger; we input the test documents here but only
    # minimize the validation error
    val_accuracy, _ = task_utils.train_and_evaluate_pos(
        train_subset, train_labels_subset, X_val, y_val)

    # we minimize the error; the lower the better
    error = 1 - float(val_accuracy)
    return error


def objective_function_pos_bilstm(feature_weights):
    """
    The objective function to optimize for POS tagging.
    :param feature_weights: a numpy array; these are the weights of the features
                            that we want to learn
    :return: the error that should be minimized
    """
    train_subset, train_labels_subset = task_utils.get_data_subsets(
        feature_values, feature_weights, X_train, y_train, POS_BILSTM,
        TASK2TRAIN_EXAMPLES[POS_BILSTM])

    # train and evaluate the tagger; we input the test documents here but only
    # minimize the validation error
    val_accuracy, _ = task_utils.train_and_evaluate_pos_bilstm(
        train_subset, train_labels_subset, X_val, y_val)

    # we minimize the error; the lower the better
    error = 1 - float(val_accuracy)
    return error


def objective_function_parsing(feature_weights):
    """
    The objective function to optimize for dependency parsing.
    :param feature_weights: a numpy array; these are the weights of the features
                            that we want to learn
    :return: the error that should be minimized
    """
    train_subset, train_labels_subset = task_utils.get_data_subsets(
        feature_values, feature_weights, X_train, y_train, PARSING,
        TASK2TRAIN_EXAMPLES[PARSING])
    val_accuracy, _ = task_utils.train_and_evaluate_parsing(
        train_subset, train_labels_subset, X_val, y_val,
        parser_output_path=parser_output_path,
        perl_script_path=perl_script_path)
    error = 100 - float(val_accuracy)
    return error


# get the task-specific methods and hyper-parameters

# num_train_examples = TASK2TRAIN_EXAMPLES[args.task]
# task_trg_domains = TASK2DOMAINS[args.task]
# read_data = data_utils.task2read_data_func(args.task)
# train_and_evaluate = task_utils.task2train_and_evaluate_func(args.task)
# objective_function = task2_objective_function(args.task)

feature_set_names = ['similarity', 'topic_similarity']
feature_names = features.get_feature_names(feature_set_names)

def convert_to_listoflisttoken(text_full_list):
    list_of_list_of_tokens = []
    for sent in text_full_list:
        if str(sent) != 'nan':
            # print('-------sent: ', sent)
            try:
                list_of_tokens = sent.split()
                # print('==========list_of_tokens: ', list_of_tokens)
                list_of_list_of_tokens.append(list_of_tokens)
            except:
                print('-------sent cannot be split: ', sent)
    return list_of_list_of_tokens


data = 'xfact'
model_dir = 'models/'+ str(data) + '/'
in_domain_train = pd.read_csv('data/'+ str(data) + '/InDomain_train.csv')
in_domain_train_list = in_domain_train['text']
in_domain_train_list_list = convert_to_listoflisttoken(in_domain_train_list)

in_domain_test = pd.read_csv('data/'+ str(data) + '/InDomain_test.csv')
in_domain_test_list = in_domain_test['text']
in_domain_test_list_list = convert_to_listoflisttoken(in_domain_test_list)

in_domain_dev = pd.read_csv('data/'+ str(data) + '/InDomain_dev.csv')
in_domain_dev_list = in_domain_dev['text']
in_domain_dev_list_list = convert_to_listoflisttoken(in_domain_dev_list)

OOD1_test = pd.read_csv('data/'+ str(data) + '/OOD1_test.csv')
OOD1_test_list = OOD1_test['text']
OOD1_test_list_list = convert_to_listoflisttoken(OOD1_test_list)

OOD2_test = pd.read_csv('data/'+ str(data) + '/OOD2_test.csv')
OOD2_test_list = OOD2_test['text']
OOD2_test_list_list = convert_to_listoflisttoken(OOD2_test_list)



text_full_list = in_domain_train_list + in_domain_test_list



os.makedirs(model_dir, exist_ok=True)

# create the vocabulary or load it if it was already created
vocab_path = os.path.join(model_dir, 'vocab.txt')
print('vocab_path: ', vocab_path)

vocab = data_utils.Vocab(20000, vocab_path) # two functions, load and create

print(vocab)




list_of_list_of_tokens = in_domain_train_list_list + in_domain_test_list_list + OOD1_test_list_list + OOD2_test_list_list

# print(list_of_list_of_tokens)
vocab.create(list_of_list_of_tokens, lowercase=True)

'''
if not os.path.exists(vocab_path):
    # retrieve all available tokenised sentences
    tokenised_sentences = data_utils.get_all_docs(
        domain2data.items(), unlabeled=True)[0]
    if args.task == PARSING:
        # get the word form from every ConllEntry
        tokenised_sentences = [[token.form if isinstance(token, ConllEntry)
                                else token for token in tokens]
                               for tokens in tokenised_sentences]
    vocab.create(tokenised_sentences)
    del tokenised_sentences
else:
    vocab.load()
'''
# load word vectors if we are using them
# word2vec = None
# if args.word2vec_path:
#     vocab_word2vec_file = os.path.join(args.model_dir, 'vocab_word2vec.txt')
#     word2vec = similarity.load_word_vectors(
#         args.word2vec_path, vocab_word2vec_file, vocab.word2id,
#         vector_size=args.vector_size, header=args.header)

# perform the task-specific pre-processing
# if args.task == SENTIMENT:
#     print('Creating binary training data...')
#     domain2train_data = data_utils.get_tfidf_data(domain2data, vocab)
# elif args.task in [POS, POS_BILSTM]:
#     print('Using words as training data for POS tagging...')
#     domain2train_data = domain2data
# elif args.task == PARSING:
#     print('Using CoNLL entries as training data for parsing. Using word '
#           'forms to extract feature representations...')
#     domain2train_data = copy.deepcopy(domain2data)
#     for domain, domain_data in domain2data.items():
#         domain_data[0] = [[conll_entry.form for conll_entry in conll_entries]
#                           for conll_entries in domain_data[0]]
# else:
#     raise ValueError('Data preproc for %s is not implemented.' % args.task)

# print('Creating relative term frequency distributions for all domains...')
term_dist_path = os.path.join(model_dir, 'term_dist.txt')

a_list_of_tokenized_documents_of_all_source_domains = [in_domain_train_list_list]
a_list_of_tokenized_documents_of_the_target_domain = [in_domain_test_list_list, OOD1_test_list_list, OOD2_test_list_list]

num_iterations = 200 # for testing, original use 2000? need to check the paper

topic_vectorizer, lda_model = similarity.train_topic_model(in_domain_train_list_list, vocab, num_topics=10, num_iterations=2, num_passes=10)

print('feature_names: ', feature_names)
feature_values = features.get_feature_representations(feature_names, in_domain_test_list_list, OOD1_test_list_list, vocab,
                                                      word2vec=None, topic_vectorizer=topic_vectorizer,
                                                      lda_model=lda_model, lowercase=True)

print('in_domain_test_list_list len: ', len(in_domain_test_list_list))
print('OOD1_test_list_list len: ', len(OOD1_test_list_list))
# get_feature_representations
# will # "get the term distribution of the entire training data
#     # (a numpy array of shape (vocab_size,) )"

# get the term distribution of each training example
# (a numpy array of shape (num_examples, vocab_size) )

# get the term distribution of the target data
# (a numpy array of shape (vocab_size, ) )

# get the topic distributions of the train and target data

print('================ feature values =======================')
print(feature_values)
print('------feature values shape, list of list-------')
print(len(feature_values))
print(len(feature_values[0]))

JSD = similarity_name2value('jensen_shannon_divergence', repr1, repr2)

"""
Retrieve the feature representations of a list of examples.
:param feature_names: a list containing the names of features to be used
:param examples: a list of tokenized documents of all source domains
:param trg_examples: a list of tokenized documents of the target domain
:param vocab: the Vocabulary object
:param word2vec: a mapping of a word to its word vector representation
                (e.g. GloVe or word2vec)
:param topic_vectorizer: the CountVectorizer object used to transform
                         tokenized documents for LDA
:param lda_model: the trained LDA model
:param lowercase: lower-case the input examples for source and target
                  domains
:return: the feature representations of the examples as a 2d numpy array of
         shape (num_examples, num_features)
"""
'''
domain2term_dist = similarity.get_domain_term_dists(term_dist_path, domain2data, vocab)

# perform optimization for every target domain
for trg_domain in args.trg_domains:
    print('Target domain:', trg_domain)

    # set the domain and similarity-specific parser output path for parsing
    parser_output_path, best_weights_parser_output_path = None, None
    if args.task == PARSING:
        parser_output_path = os.path.join(
            args.parser_output_path, '%s-%s' % (trg_domain, '_'.join(
                args.feature_sets)))
        if not os.path.exists(parser_output_path):
            print('Creating %s...' % parser_output_path)
            os.makedirs(parser_output_path)
        # use a separate subfolder for the best weights
        best_weights_parser_output_path = os.path.join(parser_output_path,
                                                       'best-weights')
        if not os.path.exists(best_weights_parser_output_path):
            os.makedirs(best_weights_parser_output_path)

    # get the training data of all source domains (not the target domain)
    X_train, y_train, train_domains = data_utils.get_all_docs(
        [(k, v) for (k, v) in sorted(domain2train_data.items())
         if k != trg_domain], unlabeled=False)

    # get the unprocessed examples for extracting the feature values
    examples, y_train_check, train_domains_check = data_utils.get_all_docs(
        [(k, v) for (k, v) in sorted(domain2data.items())
         if k != trg_domain], unlabeled=False)

    # some sanity checks just to make sure the processed and the
    # unprocessed data still correspond to the same examples
    assert np.array_equal(y_train, y_train_check)
    assert len(train_domains) == len(train_domains_check),\
        'Error: %d != %d.' % (len(train_domains), len(train_domains_check))
    assert train_domains == train_domains_check, ('Error: %s != %s' % (
        str(train_domains), str(train_domains_check)))
    if args.task in [POS, POS_BILSTM, PARSING]:
        # for sentiment, we are using a sparse matrix
        X_train = np.array(X_train)
    print('Training data shape:', X_train.shape, y_train.shape)

    # train topic model if any of the features requires a topic distribution
    topic_vectorizer, lda_model = None, None
    if any(f_name.startswith('topic') for f_name in feature_names):
        # train a topic model on labeled and unlabeled data of all domains
        topic_vectorizer, lda_model = similarity.train_topic_model(
            data_utils.get_all_docs(
                domain2data.items(), unlabeled=True)[0], vocab)

    # get the feature representations of the training data
    print('Creating the feature representations for the training data. '
          'This may take some time...')
    feature_values = features.get_feature_representations(
        feature_names, examples, domain2data[trg_domain][0], vocab,
        word2vec, topic_vectorizer, lda_model)

    if args.z_norm:
        # apply z-normalisation; this is important for good performance
        print('Z-normalizing features...')
        print('First five example features before normalisation:',
              feature_values[:5, :])
        print('Standard deviation of features:', np.std(feature_values,
                                                        axis=0))
        print('Mean of features:', np.mean(feature_values, axis=0))
        feature_values = stats.zscore(feature_values, axis=0)

    # delete unnecessary variables to save space
    del examples, y_train_check, train_domains_check

    # run num_runs iterations of the optimization and baselines in order to
    # compute statistics around mean/variance; things that vary between
    # runs: validation/test split; train set of random baseline;
    # final BayesOpt parameters; the feature values are constant for each
    # run, which is why we generate them before to reduce the overhead
    run_dict = {method: [] for method in BASELINES + [BAYES_OPT]}
    for i in range(args.num_runs):
        print('\nTarget domain %s. Run %d/%d.' % (trg_domain, i+1,
                                                  args.num_runs))

        # get the evaluation data from the target domain
        X_test, y_test, _ = domain2train_data[trg_domain]

        # split off a validation set from the evaluation data
        X_test, X_val, y_test, y_val = train_test_split(
            X_test, y_test, test_size=100, stratify=y_test
            if args.task == SENTIMENT else None)
        print('# of validation examples: %d. # of test examples: %d.'
              % (len(y_val), len(y_test)))

        # train the model with pre-learned feature weights if specified
        if args.feature_weights_file:
            print('Training with pre-learned feature weights...')
            task_utils.train_pretrained_weights(
                feature_values, X_train, y_train, train_domains,
                num_train_examples, X_val, y_val, X_test, y_test,
                trg_domain, args, feature_names, parser_output_path,
                perl_script_path)
            continue

        for baseline in args.baselines:

            # select the training data dependent on the baseline
            if baseline == RANDOM:
                print('Randomly selecting examples...')
                train_subset, _, labels_subset, _ = train_test_split(
                    X_train, y_train, train_size=num_train_examples,
                    stratify=y_train if args.task == SENTIMENT else None)
            elif baseline == ALL_SOURCE_DATA:
                print('Selecting all source data examples...')
                train_subset, labels_subset = X_train, y_train
            elif baseline == MOST_SIMILAR_DOMAIN:
                print('Selecting examples from the most similar domain...')
                most_similar_domain = similarity.get_most_similar_domain(
                    trg_domain, domain2term_dist)
                train_subset, labels_subset, _ = domain2train_data[
                    most_similar_domain]
                train_subset, _, labels_subset, _ = train_test_split(
                    train_subset, labels_subset, train_size=num_train_examples,
                    stratify=labels_subset if args.task == SENTIMENT else None)
            elif baseline == MOST_SIMILAR_EXAMPLES:
                print('Selecting the most similar examples...')
                one_all_weights = np.ones(len(feature_names))
                one_all_weights[1:] = 0
                train_subset, labels_subset = task_utils.get_data_subsets(
                    feature_values, one_all_weights, X_train, y_train,
                    args.task, num_train_examples)
            else:
                raise ValueError('%s is not a baseline.' % baseline)

            # train the baseline
            val_accuracy, test_accuracy = train_and_evaluate(
                train_subset, labels_subset, X_val, y_val,
                X_test, y_test, parser_output_path=parser_output_path,
                perl_script_path=perl_script_path)
            run_dict[baseline].append((val_accuracy, test_accuracy))

        # define the lower and upper bounds of the input space [-1, 1]
        lower = np.array(len(feature_names) * [-1])
        upper = np.array(len(feature_names) * [1])
        print('Lower limits shape:', lower.shape)
        print('Upper limits shape:', upper.shape)

        print('Running Bayesian Optimization...')
        res = bayesian_optimization(objective_function, lower=lower,
                                    upper=upper,
                                    num_iterations=args.num_iterations)

        best_feature_weights = res['x_opt']
        print('Best feature weights', best_feature_weights)
        train_subset, labels_subset = task_utils.get_data_subsets(
            feature_values, best_feature_weights, X_train, y_train,
            args.task, num_train_examples)
        val_accuracy, test_accuracy = train_and_evaluate(
            train_subset, labels_subset, X_val, y_val, X_test, y_test,
            parser_output_path=best_weights_parser_output_path,
            perl_script_path=perl_script_path)
        run_dict[BAYES_OPT].append((val_accuracy, test_accuracy,
                                      best_feature_weights))

    # log the results of all methods to the log file
    data_utils.log_to_file(args.log_file, run_dict, trg_domain, args)
'''
#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
This is the "nlpTools" module.

It can be used to perform text analysis on data to find out in
which class belongs a text sample.
"""

import numpy as np
import pandas as pd
import pickle
import cPickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.metrics import accuracy_score, confusion_matrix
import nltk

#nltk.download("stopwords")
import nltk.data
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import re
import random

# A custom stoplist
STOPLIST = set(stopwords.words('english') + ["n't", "'s", "'m", "ca"] + list(ENGLISH_STOP_WORDS))
# List of symbols we don't care about
SYMBOLS = " ".join(string.punctuation)


# Every step in a pipeline needs to be a "transformer".
# Define a custom transformer to clean text using spaCy
class CleanTextTransformer(TransformerMixin):
    """
    Convert text to cleaned text
    """

    def transform(self, X, **transform_params):
        return [clean_text(text) for text in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}


# A custom function to clean the text before sending it into the vectorizer
def clean_text(text):
    """
    The function takes some text as input and performs
    some cleaning in it.

    It returns the text file cleaned.
    """
    # print(type(text))
    # get rid of newlines
    # print text
    text = text.strip().replace("\n", " ").replace("\r", " ")

    # replace twitter @mentions
    mention_finder = re.compile(r"@[a-z0-9_]{1,15}", re.IGNORECASE)
    text = mention_finder.sub("@MENTION", text)

    # print(type(text))
    # replace HTML symbols
    text = text.replace("&amp;", "and").replace("&gt;", ">").replace("&lt;", "<")
    text = text.replace("&xbb", ">>").replace("&xab", "<<")

    # replace the nan that appears here and there
    text = text.replace(" nan ", "").replace(" nan.", "").replace("  nan", "")
    text = text.replace(",", " ")

    # replace french accent
    # text = text.encode('utf-8')
    # text = text.replace("\xe8","e").replace("\xe9","e").replace("\xe3","a")
    text = text.replace("é", "e").replace("è", "e").replace("à", "a")

    # convert special german character to "normal" on like ß -> ss, ü -> ue...
    # text = text.replace("\xe4","ae").replace("\xc4","ae")
    # text = text.replace("\xe4","ae").replace("\xc4","ae")
    text = text.replace("ä", "ae").replace("Ä", "ae").replace("&auml;", "ae").replace("&auml", "ae")
    text = text.replace("Ã¤", "ae")
    text = text.replace("[", "").replace("]", "")
    text = text.replace(" (", "").replace(") ", "")
    text = text.replace(")", " ").replace("()", " ")
    text = text.replace("/", " ")
    text = text.replace("'", " ").replace("'n", " n").replace(" &", " ")

    # text = text.replace("\xf6","oe").replace("\xd6","oe")
    text = text.replace("ö", "oe").replace("Ö", "oe")
    # text = text.replace("ö","oe")

    # text = text.replace("\xfc","ue").replace("\xdc","ue")
    text = text.replace("ü", "ue").replace("Ü", "ue")

    # text = text.replace("\xdf","ss").replace("\xb0","")
    text = text.replace("ß", "ss").replace("°", "")

    text = text.replace("`", "").replace("´", "")

    # lowercase
    text = text.lower()

    return text


def tokenize_text(sample):
    """
    A custom function to tokenize the text using spaCy
    and convert to lemmas.
    """
    # get the tokens using spaCy
    tokens = nlp_de(sample)  # parser(sample)

    # lemmatize
    lemmas = []
    for tok in tokens:
        lemmas.append(tok.lemma_.lower().strip() if tok.lemma_ != "-PRON-" else tok.lower_)
    tokens = lemmas

    # stoplist the tokens
    tokens = [tok for tok in tokens if tok not in STOPLIST]

    # stoplist symbols
    tokens = [tok for tok in tokens if tok not in SYMBOLS]

    # remove large strings of whitespace
    while "" in tokens:
        tokens.remove("")
    while " " in tokens:
        tokens.remove(" ")
    while "\n" in tokens:
        tokens.remove("\n")
    while "\n\n" in tokens:
        tokens.remove("\n\n")
    while "nan" in tokens:
        tokens.remove("nan")
    while ".." in tokens:
        tokens.remove("..")
    # print(tokens)
    return tokens


def tokenize_text_en(sample):
    """
    A custom function to tokenize the text using spaCy
    and convert to lemmas.
    """
    # get the tokens using spaCy
    tokens = nlp_en(sample)  # parser(sample)

    # lemmatize
    lemmas = []
    for tok in tokens:
        lemmas.append(tok.lemma_.lower().strip() if tok.lemma_ != "-PRON-" else tok.lower_)
    tokens = lemmas

    # stoplist the tokens
    tokens = [tok for tok in tokens if tok not in STOPLIST]

    # stoplist symbols
    tokens = [tok for tok in tokens if tok not in SYMBOLS]

    # remove large strings of whitespace
    while "" in tokens:
        tokens.remove("")
    while " " in tokens:
        tokens.remove(" ")
    while "\n" in tokens:
        tokens.remove("\n")
    while "\n\n" in tokens:
        tokens.remove("\n\n")
    while "nan" in tokens:
        tokens.remove("nan")
    while ".." in tokens:
        tokens.remove("..")
    # print(tokens)
    return tokens


def tokenize_text_de(sample):
    """
    A custom function to tokenize the text using spaCy
    and convert to lemmas.
    """
    # get the tokens using spaCy
    tokens = nlp_de(sample)  # parser(sample)

    # lemmatize
    lemmas = []
    for tok in tokens:
        lemmas.append(tok.lemma_.lower().strip() if tok.lemma_ != "-PRON-" else tok.lower_)
    tokens = lemmas

    # stoplist the tokens
    tokens = [tok for tok in tokens if tok not in STOPLIST]

    # stoplist symbols
    tokens = [tok for tok in tokens if tok not in SYMBOLS]

    # remove large strings of whitespace
    while "" in tokens:
        tokens.remove("")
    while " " in tokens:
        tokens.remove(" ")
    while "\n" in tokens:
        tokens.remove("\n")
    while "\n\n" in tokens:
        tokens.remove("\n\n")
    while "nan" in tokens:
        tokens.remove("nan")
    while ".." in tokens:
        tokens.remove("..")
    # print(tokens)
    return tokens


def printNMostInformative(vectorizer, clf, N):
    """
    Prints features with the highest coefficient values, per class
    """
    feature_names = vectorizer.get_feature_names()  # min_df=1)
    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
    topClass1 = coefs_with_fns[:N]
    topClass2 = coefs_with_fns[:-(N + 1):-1]
    print("Class 1 best: ")
    for feat in topClass1:
        print(feat)
    print("Class 2 best: ")
    for feat in topClass2:
        print(feat)

        # print(feature_names)

'''
def fun_display_category_distribution(unique_category_count, unique_category_name, limit_size_for_info=1000):
    """
    The function display the number of item per category with an arrow pointing to
    some of the category names.
    """

    # display the categories
    plt.figure()
    limit_size_for_training = 1000
    vec_pos = np.linspace(np.max(unique_category_count), 500, 50)
    aa = np.argsort(unique_category_count)
    aa = aa[::-1]
    plt.plot(np.asarray(unique_category_count)[aa], '.:')
    plt.hlines(limit_size_for_training, 0, len(aa))
    for ii in np.arange(len(aa)):
        if unique_category_count[aa[ii]] > limit_size_for_training:
            plt.annotate(unique_category_name[aa[ii]], xy=(ii, unique_category_count[aa[ii]]),
                         xytext=(len(aa) / 2, vec_pos[ii]),
                         arrowprops=dict(facecolor='blue', shrink=0.02, width=1))
    plt.xlim([0, len(aa)])
'''

def fun_get_unique_item_count_and_name(dataFrame, column_name, verbose = None):
    """
    The function extracts the list of unique items and their corresponding occurence.

    Args:
        dataFrame: a panda DataFrame
        column_name: a (str) corresponding to a dataFrame column name

    Returns:
        unique_count: a (list of int)
        unique_name: a (list of str)
    """

    unique_count = []
    unique_name = list(dataFrame.loc[:, column_name].unique())

    for cc in unique_name:
        unique_count.append(len(dataFrame[dataFrame[column_name] == cc]))

        if verbose != None:
            print "Category <%s> has %1.0f item. " % (cc, unique_count[-1])

    print "There are <%1.f> different categories in the column [%s]." % (len(unique_name), column_name)

    return unique_name, unique_count

#
# Here we select the data for training and testing our system.
#

def fun_create_train_and_test_data_old1(dataFrameFashion,
                                        limit_number_item=20000,
                                        number_train_data=50,
                                        number_test_data=5):
    """
    The function create train and test data.
    In:
        - dataFrameFashion (pandas DataFrame) containd the data we want to analyze
        - limit_number_item: below this limit the class is not taking into account.
        - number_train_data: number of element to take for the train data.
        - number_test_data: number of test data to use for the classification evaluation.

    Example of use:
    > train, labelsTrain, test, labelsTest =
        fun_create_train_and_test_data(df, limit_number_item = 1000,
                                       number_train_data=100,
                                       number_of_test=100)
    """

    # create list of the different categories as classes
    unique_category_name = list(dataFrameFashion.iloc[:, 2].unique())
    unique_category_count = []
    for cc in unique_category_name:
        unique_category_count.append(len(dataFrameFashion[dataFrameFashion.category == cc]))

    # CAN BE BETTER WRITTEN
    train_tdcb, train_tdc, train_td, train_d, train_t = [], [], [], [], []
    labelsTrain = []
    test_tdcb, test_tdc, test_td, test_d, test_t = [], [], [], [], []
    labelsTest = []
    test_uncat_tdcb, test_uncat_tdc, test_uncat_td, test_uncat_d, test_uncat_t = [], [], [], [], []
    labelsTest_uncat = []

    nb_pre_defined_classes = 0
    for ii in np.arange(len(unique_category_name)):
        # get lenght of item per category
        if unique_category_name[ii] == 'uncategorized':
            # get list of item with this category
            df_category = dataFrameFashion[dataFrameFashion.category == 'uncategorized']
            index_category = np.asarray(df_category.index)
            # print len(index_category)

            index_item_for_training = np.arange(len(index_category))

            #    len(inderandom.sample(range(0, unique_category_count[ii]),
            #                                            10000)# number of sample data

            for jj in np.arange(len(index_item_for_training)):
                title = unicode(str(df_category.iloc[index_item_for_training[jj], 5]), 'utf-8')
                description = unicode(str(df_category.iloc[index_item_for_training[jj], 6]), 'utf-8')
                colour = unicode(str(df_category.iloc[index_item_for_training[jj], 8]), 'utf-8')
                feedCategory = unicode(str(df_category.iloc[index_item_for_training[jj], 3]), 'utf-8')
                brand = unicode(str(df_category.iloc[index_item_for_training[jj], 7]), 'utf-8')

                test_uncat_tdcb.append(title + '. ' + description + '. ' + colour + '. ' + brand)
                test_uncat_tdc.append(title + '. ' + description + '. ' + colour)
                test_uncat_td.append(title + '. ' + description)
                test_uncat_d.append(description)
                test_uncat_t.append(title)
                labelsTest_uncat.append('uncategorized')

        else:
            if unique_category_count[ii] > limit_number_item:
                # get list of item with this category
                df_category = dataFrameFashion[dataFrameFashion.category == unique_category_name[ii]]
                index_category = np.asarray(df_category.index)

                # ze the selection of data for training and testing
                index_item_for_training = random.sample(range(0, unique_category_count[ii]),
                                                        number_train_data + number_test_data)

                # data for training
                for jj in np.arange(number_train_data):
                    title = unicode(str(df_category.iloc[index_item_for_training[jj], 5]), 'utf-8')
                    description = unicode(str(df_category.iloc[index_item_for_training[jj], 6]), 'utf-8')
                    colour = unicode(str(df_category.iloc[index_item_for_training[jj], 8]), 'utf-8')
                    feedCategory = unicode(str(df_category.iloc[index_item_for_training[jj], 3]), 'utf-8')
                    brand = unicode(str(df_category.iloc[index_item_for_training[jj], 7]), 'utf-8')

                    train_tdcb.append(title + '. ' + description + '. ' + colour + '. ' + brand)
                    train_tdc.append(title + '. ' + description + '. ' + colour)
                    train_td.append(title + '. ' + description)
                    train_d.append(description)
                    train_t.append(title)
                    labelsTrain.append(unique_category_name[ii])

                # data for testing
                for jj in np.arange(number_train_data, len(index_item_for_training)):
                    title = unicode(str(df_category.iloc[index_item_for_training[jj], 5]), 'utf-8')
                    description = unicode(str(df_category.iloc[index_item_for_training[jj], 6]), 'utf-8')
                    colour = unicode(str(df_category.iloc[index_item_for_training[jj], 8]), 'utf-8')
                    feedCategory = unicode(str(df_category.iloc[index_item_for_training[jj], 3]), 'utf-8')
                    brand = unicode(str(df_category.iloc[index_item_for_training[jj], 7]), 'utf-8')

                    test_tdcb.append(title + '. ' + description + '. ' + colour + '. ' + brand)
                    test_tdc.append(title + '. ' + description + '. ' + colour)
                    test_td.append(title + '. ' + description)
                    test_d.append(description)
                    test_t.append(title)
                    labelsTest.append(unique_category_name[ii])

                nb_pre_defined_classes = nb_pre_defined_classes + 1
                del df_category

    labelNames = ['title description color brand', 'title description color', 'title description', 'description',
                  'title']
    train = [train_tdcb, train_tdc, train_td, train_d, train_t]
    labelsTrain = labelsTrain
    test = [test_tdcb, test_tdc, test_td, test_d, test_t]
    labelsTest = labelsTest
    test_uncat = [test_uncat_tdcb, test_uncat_tdc, test_uncat_td, test_uncat_d, test_uncat_t]
    print "We have %1.0f classes for training with %1.0f element each and %1.0f test data." % (nb_pre_defined_classes,
                                                                                               number_train_data,
                                                                                               number_test_data)

    return train, labelsTrain, test, labelsTest, test_uncat, labelsTest_uncat, labelNames


def fun_create_train_and_test_data(dataFrame,
                                   columns_name,
                                   number_minimum_item=40000,
                                   number_train_data=5,
                                   number_test_data=5):
    """
    The function create train and test data.
    In:
        - dataFrame (pandas DataFrame) containd the data we want to analyze
        - columns_name (list str): a combination between ['title','description','colour','brand']
        - number_minimum_item: below this limit the class is not taking into account.
        - number_train_data: number of element to take for the train data.
        - number_test_data: number of test data to use for the classification evaluation.

    Out:
        - train
        - labelsTrain
        - test
        - labelsTest
        - test_uncat
        - labelNames



    Example of use:
    > train, labelsTrain, test, labelsTest = fun_create_train_and_test_data(df, ['title','colour'])

    """

    # create list of the different categories as classes with the length
    unique_category_name = list(dataFrame.loc[:, 'category'].unique())
    unique_category_count = []
    for cc in unique_category_name:
        unique_category_count.append(len(dataFrame[dataFrame.category == cc]))

    # train_, labelsTrain, test_, labelsTest, test_uncat_ = [], [], [], []

    df_select = dataFrame[dataFrame.category == 'uncategorized']
    index_select = np.asarray(df_select.index)
    index_col = ['title', 'colour']
    df_test_uncat_ = pd.concat([dataFrame.iloc[index_select, index_col]])
    del df_select

    df_test_uncat_['title'] = df_test_uncat_['title'].str.cat(df_test_uncat_['colour'].values.astype(str), sep=' >< ')
    df_test_uncat_ = df_test_uncat_.drop(df_test_uncat_.columns[[1]], axis=1)

    return list(df_test_uncat_.iloc[:, 0])


def fun_extract_all_categories(dataFrame, column_names,
                               unique_category_name, unique_category_count,
                               limit_number_item=10000,
                               number_train_data=5,
                               number_test_data=5):
    """
    Whatever the parameters the function will always extract all the uncategorized
    element.

    It returns data as lists.
    """

    # initialization:
    list_cat, list_label, tr_, labelsTr_, te_, labelsTe_ = [], [], [], [], [], []
    te_uncat_, labelsTe_uncat_ = [], []

    # looptroop is aswedish hip-hop brand
    for name_cat, count_cat in zip(unique_category_name, unique_category_count):
        if name_cat == "uncategorized":
            te_uncat_, labelsTe_uncat_ = fun_extract_category(dataFrame, name_cat, column_names)
        else:  # count_cat > limit_number_item:
            # now check how many data we want
            if count_cat > limit_number_item:
                # then it is any other categories, we extract all
                list_data, list_label = fun_extract_category(dataFrame, name_cat, column_names)

                # print "working on %s category." % name_cat
                # print "data of size %1.0f." % len(list_data)

                # get random index
                # print "diff info cat %1.0f" % (len(list_data) - count_cat)
                ind_ = random.sample(xrange(count_cat), number_test_data + number_train_data)

                # print ind_
                for kk in np.arange(0, number_train_data):
                    # print ind_[kk], list_label[ind_[kk]]
                    tr_.append(list_data[ind_[kk]])
                    labelsTr_.append(list_label[ind_[kk]])

                for ll in np.arange(number_train_data, len(ind_)):
                    # print ind_[ll], list_label[ind_[ll]]
                    te_.append(list_data[ind_[ll]])
                    labelsTe_.append(list_label[ind_[ll]])

                del list_data, list_label

    return list_cat, tr_, labelsTr_, te_, labelsTe_, te_uncat_


def fun_extract_category(dataFrame, category_selection, column_names):
    """
    The function extracts element from the dataFrame selecting the chosen
    columns by reading the category_selection:
    IN:
        - dataFrame (pandas dataFrame): tha dataFrame containing the data.
        - category_selection (char): the name of the chosen category
        - column_names (chart): a list with the name of the columns to merge

    OUT:
        - a list of the combined columns
    """

    # select index value for the given category selected
    df_select = dataFrame[dataFrame.new_category == category_selection]
    # df_select = dataFrame[dataFrame.category == category_selection]
    index_select = np.asarray(df_select.index)

    index_col = np.zeros(len(column_names))
    for i, col in enumerate(column_names):
        index_col[i] = list(dataFrame.columns).index(col)

    df_test_ = pd.concat([dataFrame.iloc[index_select, index_col]])
    del dataFrame

    # merge two to n columns
    ## unicode(str(df_category.iloc[index_item_for_training[jj],7]), 'utf-8')

    df_test_['dataText'] = df_test_[column_names].apply(lambda x: ' '.join(x.map(str)), axis=1)
    list_category = list(df_test_.iloc[:, -1])
    list_label = list([category_selection] * len(list_category))

    return list_category, list_label


def fun_do_classification(train, labelsTrain, test, labelsTest, corpus_lang="en", param_ngram_range=(1, 1),
                          param_token_pattern='(?u)\\b\\w\\w+\\b'):
    """
    It returns the global score of the experimentation.

    The experimentation is performed for EN and DE tokenizer.

    Optionnal parameters:
         - corpus_lang is "en" by default but "de" can be chosen, it will influences
         the choise of the language for the Spacy tokenizer.
         - param_ngram_range=(1,1)                  | if (1,n) for n grams
         - param_token_pattern= '(?u)\\b\\w\\w+\\b' | with this -> '(?u)\\b\\w+\\b'

    And more.

    """

    # the vectorizer and classifer to use with the righ language
    if corpus_lang == "de":
        vectorizer = CountVectorizer(tokenizer=tokenize_text_de,
                                     ngram_range=param_ngram_range,
                                     token_pattern=param_token_pattern)
    elif corpus_lang == "en":
        vectorizer = CountVectorizer(tokenizer=tokenize_text_en,
                                     ngram_range=param_ngram_range,
                                     token_pattern=param_token_pattern)
    else:
        vectorizer = CountVectorizer(tokenizer=word_tokenize,
                                     ngram_range=param_ngram_range,
                                     token_pattern=param_token_pattern)

    # choose classifier
    clf = LinearSVC()
    pipe = Pipeline([('cleanText', CleanTextTransformer()),
                     ('vectorizer', vectorizer),
                     ('clf', clf)])
    # define the pipe
    pipe.fit(train, labelsTrain)

    # do prediction
    preds_all = pipe.predict(test)

    # compute the performances
    global_accuracy_score = accuracy_score(labelsTest, preds_all)

    return global_accuracy_score, preds_all


def fun_display_global_accuracy(global_accuracy, labelSession):
    """
    Function to display the global result of an experimentation with classes of different sizes.
    """
    # plt.figure()

    for i, labelS in enumerate(labelSession):
        vec_x = np.arange(1, np.shape(global_accuracy)[0] + 1)
        vec_y = global_accuracy[:, i]
        # print vec_x, vec_y
        plt.plot(vec_x, vec_y, '+:', markersize=10, label=labelS)
    plt.legend(loc=3)
    plt.ylim(0, 1.1)
    plt.xlim(0, len(vec_x) + 1)
    plt.xlabel('Type of train data')
    plt.ylabel('Accuracy')
    plt.draw()


# fun_display_global_accuracy(np.random.random((5,2)),['description', 'description and title'])

def fun_save_all_description_as_text_old1(dataFrame, index_dataFrame, limit_number_item=10,
                                          file_name="sample_data.txt"):
    """
    The function takes as input a dataFrame (e.g. the one we use after importing
    the big csv file) and the index_dataFrame corresponding to the dataFrame
    columns we want to save as text data.
    """

    print "There file is made of %1.0f items" % limit_number_item

    # open file for writing
    f = open(file_name, 'w')
    for ii in np.arange(limit_number_item):
        word_data = ''
        for cc in index_dataFrame:
            cell_data = unicode(str(dataFrame.iloc[ii, cc]), 'utf-8')
            word_data = word_data + ' ' + cell_data + '\n'
            # print word_data
            # clean word_data?
        word_data = clean_text(word_data)
        # print word_data

        f.write(word_data)
        del word_data
    f.close()


def save_category_data_as_text(dataFrame, category_name, column_names):
    """
    The function reads the dataFrame and store each line having category name
    as index to a text file. Also it can choose which columns we want to merge.
    IN:
        - dataFrame (pandas dataframe)
        - category (str): e.g. male.tshirt.shortsleeve
        - columns_name (list str): e.g. ['title','description','colour']
    OUT:
        - nothing is return, only a text file is saved under the name
        category_####.txt

    Example of use:
    fun_save_category_data_as_text(dataFrame, 'male.tshirt.shortsleeve', ['title','description'])
    """

    # get a list of the data before saving as text
    list_category, list_label = fun_extract_category(dataFrame, category_name, column_names)

    file_name = ''
    for column in column_names:
        file_name = file_name + column[0]

    file_name = category_name + '_' + file_name + '_'
    f = open(file_name + '.txt', 'w')
    # -> get all data from a category_name
    f.writelines(["%s\n" % item for item in list_category])
    f.close()

    # save as pickle
    fp = open(file_name + '.pickle', 'wb')
    pickle.dump(list_category, fp)
    fp.close()

    # save as json???


def save_all_description_as_text(dataFrame, index_dataFrame, limit_number_item=10, file_name="sample_data.txt"):
    """
    The function takes as input a dataFrame (e.g. the one we use after importing
    the big csv file) and the index_dataFrame corresponding to the dataFrame
    columns we want to save as text data.
    """

    print "There file is made of %1.0f items" % limit_number_item

    # open file for writing
    f = open(file_name, 'w')
    for ii in np.arange(limit_number_item):
        word_data = ''
        for cc in index_dataFrame:
            cell_data = unicode(str(dataFrame.iloc[ii, cc]), 'utf-8')
            word_data = word_data + ' ' + cell_data + '\n'
            # print word_data
            # clean word_data?
        word_data = clean_text(word_data)
        # print word_data

        f.write(word_data)
        del word_data
    f.close()


def run_classification_with_spacy(train, labelsTrain, test, labelsTest, test_uncat, param_ngram_range, corpus_lang):
    """
    The function takes as input the parameters to perform the classification
    using spacy.

    IN:
        - train (str): the decription of items.
        - labelsTrain (str): the category of each item.
        - test (str): the descritopm of test items.
        - labelsTest (str): the category of each test item.
        - test_uncat (str): the description of uncategorized items.
        - param_ngram_range (float): (1,n) for n-gram
        - corpus_lang (str); : "en" or "de" to select the rigth Spay tokenizer.

    Out:
        - preds_all (str): the category prediction for each test item
        - global_accuracy_score (float): the global score on all the prediction
        - conf_matrix ()
    """
    # param for the n-gram approach
    if param_ngram_range[1] > 1:
        param_token_pattern = '(?u)\\b\\w+\\b'
    else:
        param_token_pattern = '(?u)\\b\\w\\w+\\b'

    if corpus_lang == "de":
        vectorizer = CountVectorizer(tokenizer=tokenize_text_de,
                                     ngram_range=param_ngram_range,
                                     token_pattern=param_token_pattern)
    else:
        vectorizer = CountVectorizer(tokenizer=tokenize_text_en,
                                     ngram_range=param_ngram_range,
                                     token_pattern=param_token_pattern)

    # choose classifier
    clf = LinearSVC()
    pipe = Pipeline([('cleanText', CleanTextTransformer()),
                     ('vectorizer', vectorizer),
                     ('clf', clf)])

    # define the pipe
    pipe.fit(train, labelsTrain)

    # do prediction
    preds_all = pipe.predict(test)

    # compute the performances
    global_accuracy_score = accuracy_score(labelsTest, preds_all)

    # computer confusion matrix
    conf_matrix = confusion_matrix(preds_all, labelsTest)

    # do prediction of uncat data:
    preds_uncat = pipe.predict(test_uncat)

    return preds_all, global_accuracy_score, conf_matrix, preds_uncat


def run_classification(train, labelsTrain, test, labelsTest, test_uncat, ngram_range_val=(1, 1),
                       tokenizer_mother_tongue='simple'):
    """
    The function takes as input the parameters to perform the classification
    using spacy(en/de), nltk or word_tokenize

    IN:
        - train (str): the decription of items.
        - labelsTrain (str): the category of each item.
        - test (str): the descritopm of test items.
        - labelsTest (str): the category of each test item.
        - test_uncat (str): the description of uncategorized items.
        - param_ngram_range (float): (1,n) for n-gram
        - corpus_lang (str); : "en" or "de" to select the rigth Spay tokenizer.

    Out:
        - preds_all (str): the category prediction for each test item
        - global_accuracy_score (float): the global score on all the prediction
        - conf_matrix ()
    """
    # param for the n-gram approach
    strip_accents_val = 'unicode'
    vocabulary_val = None
    # ngram_range_val = (1,1)
    # tokenizer_mother_tongue = "simple"
    # min_df_val = 1#50 # remove item that appear at minimum in 15 documents
    # max_df_val = 1#0.75 # remove item that appear in more than 75% of the documents

    # param_token_pattern = '(?u)\\b\\w+\\b'
    if ngram_range_val[1] > 1:
        param_token_pattern = '(?u)\\b\\w+\\b'
    else:
        param_token_pattern = '(?u)\\b\\w\\w+\\b'

    # choice of the tokenizer
    if tokenizer_mother_tongue == "DE":
        tokenizer_name = "DE_spacy"
        vectorizer = CountVectorizer(tokenizer=tokenize_text_de,
                                     ngram_range=ngram_range_val,
                                     token_pattern=param_token_pattern)
        ##input='filename',
        # tokenizer=tokenizeText_de,
        # min_df = min_df_val,
        # max_df = max_df_val,
        # strip_accents = strip_accents_val,
        # vocabulary = vocabulary_val,
        # ngram_range = ngram_range_val,
        # token_pattern = param_token_pattern)
    elif tokenizer_mother_tongue == "EN":
        tokenizer_name = "EN_spacy"
        vectorizer = CountVectorizer(tokenizer=tokenize_text_de,
                                     ngram_range=ngram_range_val,
                                     token_pattern=param_token_pattern)
        ##input='filename',
        # tokenizer=tokenizeText_en,
        # min_df = min_df_val,
        # max_df = max_df_val,
        # strip_accents = strip_accents_val,
        # vocabulary = vocabulary_val,
        # ngram_range = ngram_range_val,
        # token_pattern = param_token_pattern)
    elif tokenizer_mother_tongue == "nltk":
        tokenizer_name = "nltk"
        print "the choosen token_pattern is %s" % param_token_pattern
        vectorizer = CountVectorizer(analyzer='word',
                                     tokenizer=word_tokenize,
                                     ngram_range=ngram_range_val,
                                     token_pattern=param_token_pattern)  # ,
        ##input='filename',
        # tokenizer = word_tokenize,
        ##tokenizer = RegexpTokenizer('\w+|\d{2,3}\scm').tokenize,
        # min_df = min_df_val,
        # max_df = max_df_val)
        # strip_accents = strip_accents_val,
        # vocabulary = vocabulary_val)
        # ngram_range = ngram_range_val,
        # token_pattern = param_token_pattern)
    else:
        tokenizer_name = "simple"
        # here we use the tokenizer that comes with sklearn I believe
        vectorizer = CountVectorizer(  # input='filename',
            # min_df = min_df_val,
            # max_df = max_df_val,
            strip_accents=strip_accents_val,
            vocabulary=vocabulary_val,
            ngram_range=ngram_range_val,
            token_pattern=param_token_pattern)

    print "You did choose the %s tokenizer." % tokenizer_name
    # choose classifier
    clf = LinearSVC()
    pipe = Pipeline([('cleanText', CleanTextTransformer()),
                     ('vectorizer', vectorizer),
                     ('clf', clf)])

    # define the pipe
    pipe.fit(train, labelsTrain)

    # do prediction
    preds_all = pipe.predict(test)

    # compute the performances
    global_accuracy_score = accuracy_score(labelsTest, preds_all)

    # computer confusion matrix
    conf_matrix = confusion_matrix(preds_all, labelsTest)

    # do prediction of uncat data:
    preds_uncat = pipe.predict(test_uncat)

    return preds_all, global_accuracy_score, conf_matrix, preds_uncat


def give_info_dataframe_categories(dataFrame, column="category"):
    """
    The function returns the list of unique category and another list with the
    occurence of item for each category. The category "uncategorized" is also
    taken as a category.

    And if you want this info for the colour colum you do as follows:
    > unique_name, unique_count = fun_give_info_dataframe_categories(df, column = "colour")

    """
    # create list of the different categories as classes
    unique_names = list(dataFrame.loc[:, column].unique())
    unique_count = []
    for unique_name in unique_names:
        unique_count.append(len(dataFrame[dataFrame.loc[:, column] == unique_name]))

    return unique_names, unique_count


def save_list_as_text(list_of_item, name_list_of_item):
    """
    As its name describes it already pretty good, the function take as input
    a list of text (or number) and save the list as text file.

    Args:
        colour_tag: a string
    Returns:
        colour_tag: a string (unmodified if the string colour_tag doesn't belong to old_list.
    """

    file_name = name_list_of_item

    # open file for writing
    f = open(file_name + '_.txt', 'w')
    for item_list in list_of_item:
        if type(item_list) == int:
            item_list = str(item_list)
        f.write(item_list + '\n')
    f.close()

    # save as pickle
    cPickle.dump(list_of_item, open(file_name + '_.pickle', 'wb'))
    # and to read back -> list_of_item = cPickle.load(open(file_name+'_.pickle', 'rb'))


def fun_clean_colour_name(colour_tag):
    """
    The function is applied to the colour column of the giant dataFrame.

    Basically it applies a lower_case() function to all values and converts to 'nan' if the cell has the type float
    or int.

    Args:
        colour_tag: a string
    Returns:
        colour_tag: a string (unmodified if the string colour_tag doesn't belong to old_list.
    """

    if (type(colour_tag) == int) or (type(colour_tag) == float):
        colour_tag = "none"

    colour_tag = colour_tag.lower()
    colour_tag = re.sub(' +',' ', colour_tag) # replace multiple white space
    colour_tag = clean_text(colour_tag)

    return colour_tag


def fun_get_color_name_reduce(colour_tag):
    """
    The function takes a color name as input and return a reduced value of it in English.

    To do so it uses regex to find out if the EN or DE color name is present.

    e.g. the colour word "hellblau" will return "blue".

    We still need to do something for color name combination such as "red / blue".
    """

    colour_tag_out = colour_tag

    list_base_colors_en = ['red',
                           'pink',
                           'violet',
                           'blue',
                           'cyan',
                           'green',
                           'yellow',
                           'orange',
                           'brown',
                           'white',
                           'grey',
                           'black']

    list_base_colors_de = ['rot',
                           'rosa',
                           'veilchen',
                           'blau',
                           'cyan',
                           'gruen',
                           'gelb',
                           'orange',
                           'braun',
                           'weiss',
                           'grau',
                           'schwarz']


    # for base_color_en, base_color_de in zip(list_base_colors_de, list_base_colors_en):
    if colour_tag != "none":
        for (base_color_en, base_color_de) in zip(list_base_colors_en, list_base_colors_de):
            pattern = base_color_en+'|'+base_color_de
            # the "search" is better than "match" as it check all over the string and not
            # only the beginning of the string in which we search the pattern.
            searchObj = re.search(pattern, colour_tag, re.M)
            if searchObj is not None:
                colour_tag_out = base_color_en
                break

    return colour_tag_out


def fun_map_color_to_new_color(colour_tag):#, old_color_list, new_color_list):
    """
    The function maps the colour tag if it belongs to the old_list to the corresponding
    item is the new_list.

    e.g. the colour_tag "navy" will be mapped to "blue"

    Args:
        colour_tag: a string
    Returns:
        colour_tag_out: a string (unmodified if the string colour_tag doesn't belong to old_list.
    """

    # where do I put NEUTRAL color???


    # where do I put tencel???

    # where do I put CHAMPAGNE color?? in yellow?

    colour_tag_out = colour_tag
    colour_tag = re.sub(' +', ' ', colour_tag)  # replace multiple white space

    old_color_list_red = ['berry brow', 'bordo', 'brick', 'burgandy marl', 'burgund', 'burgundy', 'burgundy marl',
                          'cardinal', 'cassis', 'cayenne', 'cherry', 'chili', 'corale', 'cranberry', 'crimson',
                          'dark berry', 'dk burgondy', 'dusty ceda', 'kirsche', 'korall', 'koralle', 'line syrah',
                          'lingonberry', 'marsala', 'merlot', 'mulberry', 'ocker', 'picante', 'pompeian r', 'reed',
                          'ruby', 'sassafras', 'tomate', 'tomato', 'volcano', 'wild berry', 'wild berry checks',
                          'wine',]

    old_color_list_pink = ['almond blossom', 'blush', 'cerise', 'cobbler', 'cobler dip wash', 'cobler smash', 'coral',
                           'dk aged cobler', 'echo park', 'flamingo', 'fuchsia', 'fuschia', 'lachs', 'light coral',
                           'light coral marl', 'light kecap', 'malve', 'peony', 'rhubarb', 'rose', 'rosÈ', 'rosÉ',
                           'rosé', 'rosé', 'salmon', 'schwein', 'shrimp', 'thistle', 'vintage aged cobbler',
                           'vintage aged cobler', 'watermelon']

       # blush", "fuchsia","fuschia", "shrimp", "rose", "rosÈ", "rosé", 'flamingo', "salmon",
       #                    "light coral", "light coral marl", "coral", "watermelon", "cerise",
       #                    "almond blossom","thistle","vintage aged cobler","cobbler", "'vintage aged cobbler",
       #                    "cobler dip wash","cobler smash","dk aged cobler","peony","light kecap","rhubarb",
       #                    "ros\xc3\x89","rosé","malve","echo park","lachs","schwein"]

    old_color_list_violet = ['aubergine', 'beere', 'berry purple', 'bright purple', 'dark magenta', 'dark purple',
                             'dark purple marl', 'flieder', 'grapevine', 'heliotrope purple', 'hyacinth',
                             'hyacinth checks', 'lavendel', 'lavender', 'lila', 'mauve', 'orchid', 'pflaume',
                             'plum fog', 'prune', 'purple', 'purple smoke', 'schimmernde butter', 'syringa',
                             'twilight', 'twilit', 'velvet bro', 'velvet cloud', 'wisteria']

    old_color_list_blue = ['718 royal', 'atlantic', 'atlantico', 'avio', 'azul', 'azure', 'breeze', 'chambray',
                           'coastal bl', 'cobalt', 'dark indig', 'dark indigo', 'dark rinse used', 'deep ocean',
                           'denim', 'dk pacific', 'faded flow', 'fresh breeze', 'fresh breeze checks', 'indigo',
                           'insignia b', 'intense bl', 'jeans', 'kentucky b', 'kentucky blue', 'kobalt', 'kornblume',
                           'light aged destroy', 'lt aged destroy', 'marin', 'marine', 'medium blu',
                           'medium vintage aged', 'midnight', 'mood indig', 'moon dot bl', 'moonlight', 'navy',
                           'north atlantic', 'ocean', 'old rinse', 'opaque ind', 'pacific', 'pacific check',
                           'pearcoat', 'periwinkle', 'petrol', 'rinse', 'rinse truc', 'rinse used', 'rinse washed',
                           'rinse washed', 'riviera', 'sapphire', 'sapphire bl htr', 'sea htr', 'sky', 'surf shack',
                           'turkish co', 'vintage bl', 'washed blu', 'water','super bleach washed']

    old_color_list_cyan = ["aqua", "tuerkis", "turquoise", "pale turqouise", "pale turquoise"]

    old_color_list_green = ["light gold green", "olive", "oilve", "khaki", "kaki","mint", "shaded spruce", "shaded spr",
                            "sage", "teal","pale teal", "petrol gre","verde bott", "jade", "cactus bla", "aloe",
                            "dk moss ao","mdf ao","pine","forest", "four leaf","cactus","peridot","dried herb","dusty gree",
                            "sycamore","dark moss","dk moss combat ao","dk moss","emerald gr", "garden",
                            "seagrass","bright absinth","chartreuse","tropenfrucht","ginko","apfel","grape leaf",
                            "pistachio dark moss","smaragd","june bug m","aroch","spruce","emeraude","moss",
                            "dark moss melange"]

    old_color_list_yellow =  ['champagne', 'champan fa', 'citron', 'curcuma', 'dark gold', 'egg shell', 'el prato',
                              'gold', 'golden lea', 'honey', 'honig', 'lemon', 'light sesame', 'lime', 'mandarin',
                              'mellow yel', 'messing', 'mustard', 'pumpkin', 'sack', 'sahara', 'senf', 'sunshine',
                              'tiger', 'zitrone']

    old_color_list_orange = ['aftersun', 'amber', 'apricot', 'bakelite', 'black orange', 'blk orng', 'butterscotch',
                             'cinnamon', 'kuerbis', 'mango', 'papaya', 'rooibos', 'rust', 'sassafras marl', 'tangerine',
                             'tawny', 'terracotta']

    old_color_list_brown = ['antique bronze marl', 'arabian sp', 'bark', 'bordeaux', 'bronze', 'bronze carnelian',
                            'brune', 'cacao', 'canyon', 'caramel', 'cedar wood', 'chestn', 'chestnut', 'chocolate',
                            'cocoa', 'cognac', 'copper', 'cuero', 'cuero magn', 'dark wom', 'dark worn', 'elmwood',
                            'erdfarben', 'espresso', 'gauco pito', 'havana', 'high rise', 'kupfer', 'lohfarbe',
                            'maroon', 'milchkaffee', 'moca', 'mocca', 'mocha bisq', 'mocha shaq', 'negro', 'nutmeg',
                            'oak', 'provincial', 'rost', 'rustic bro', 'rustic bro', 'saddle', 'schoko', 'serraje wh',
                            'tabak', 'tan', 'terra', 'toffee', 'toggee', 'verona', 'walnut', 'wood thrush',
                            'wood trush', 'ziegel']

    old_color_list_white = ['bright whi', 'bright white', 'buttermilch', 'chalk whit', 'dirty whit', 'egret',
                            'eierschale', 'fleur sel', 'ftwwht whi', 'ice', 'kokosnussoel', 'light aged', 'light bone',
                            'light mole', 'lt aged', 'lt wave htr', 'major whit', 'milk', 'mole light', 'opela',
                            'pearil', 'pearl', 'perle', 'plmfg blch', 'sail sail', 'vanilla ic', 'vintage wh']

    old_color_list_grey = ['abbey ston', 'aircraft heather', 'anthracite', 'anthrazit', 'anthrazit meliert',
                           'antracite', 'asfalt', 'asfalt htr', 'ash', 'auster', 'austernfarben', 'birch', 'capour pin',
                           'carbid', 'carbon', 'carbon hea', 'cloud', 'dark aged', 'dark gun gray', 'dark shell',
                           'dark sulphur', 'dk aged', 'duffle gre', 'duffle grey', 'elefant', 'excalibur', 'fango',
                           'fearn', 'fumo', 'glacier gr', 'glacier grey', 'graphit', 'gray', 'gray', 'greige', 'gris',
                           'gunmetal', 'heather', 'heather ne', 'hielo', 'high rise', 'high rise marl', 'ikat aop',
                           'iron', 'kitt', 'lt airo', 'lt cloud', 'lt cloud htr', 'marble', 'medium aged', 'medium gre',
                           'mercury', 'metallic', 'milk ao', 'mink', 'moon rock', 'muschel', 'muskat', 'neutral',
                           'niagara', 'nomad', 'orphus', 'osaka', 'peacoat', 'pewter', 'phantom', 'platinum',
                           'port royal', 'quarz', 'quiet shad', 'quiet shadow', 'rainy day', 'raw', 'rinsed',
                           'rinsed check', 'rover', 'salbei', 'shade', 'shadow gre', 'shell', 'silber', 'silver',
                           'silver gre', 'silver semi matt', 'silver semi-mat', 'urban chic', 'wlf gry-wh']

    old_color_list_black = ['3d aged', '\bink\b', 'anker', 'black semi matte', 'cafe', 'calzo', 'caviar', 'charcoal',
                            'clean blac', 'coffee', 'combat', 'croco blac', 'dark acid', 'dark combat', 'dark fig',
                            'dark iris', 'dark rinse used', 'dark saphi', 'dark steel classic check', 'dark vintage',
                            'dk baron', 'ebony', 'mid vintage', 'monochrome mele', 'night', 'raven', 'total ecli',
                            'varsity ma']

    old_color_list_multi =   ['all over p', 'assorted', 'bottle', 'bunt', 'candy', 'clay', 'ditsy prin', 'dot off wh',
                              'dot print', 'exotic palms', 'flame', 'flame check', 'galaxy', 'gemustert', 'gestreift',
                              'holografisch', 'leave prin', 'meadow flo', 'meadow flower', 'mehrfarbig', 'metallmix',
                              'multi', 'multicolour', 'opal', 'powder', 'powder pin', 'print', 'stone', 'streifen',
                              'stripe', 'stripe 1', 'stripe 1', 'stripe 2', 'stripes', 'sugar', 'tourmaline', 'urban',
                              'waikiki tropic', 'wakiki tropic']

    old_color_list_beige =   ['beige dark', 'beige light', 'beige melange', 'biscuit', 'camel', 'camel mela',
                              'clear crea', 'clear crem', 'cork', 'cream', 'crema', 'creme', 'desert', 'ecru',
                              'elfenbein', 'fade-out beige', 'hellbeige', 'hellpuder', 'ivory', 'light beige',
                              'natur', 'new beige', 'sand', 'suede beig', 'summer beige', 'taupe', 'turtledove',
                              'vanilla', 'vanille']

    old_color_list_skintone =['doe', 'dune', 'haut', 'hautfarbe', 'ight powd', 'light tan', 'macadamia', 'nude',
                              'peach', 'pfirsich', 'powder str', 'rugby ta', 'sesame', 'skin beige', 'skintone',
                              'skintone', 'soft hazel']


    old_color_list_not_sorted = ['\b0\b', 'awaken', 'bleached a', 'braeunungsfett', 'canopy gre', 'ceniza nik',
                                 'classic ti', 'cuadros fo', 'cupcake', 'dezember', 'espanol', 'extreme painted',
                                 'eye allove', 'flower pri', 'gebleicht', 'ground', 'hafer', 'hufeisen', 'kit',
                                 'leaves pri', 'light vintage aged', 'light worn', 'little boy', 'lt wave', 'maybe',
                                 'mid worn', 'monochrome', 'moor', 'movin shak', 'nerz', 'ohne angabe', 'one colour',
                                 'original', 'plaster', 'rosin mel', 'schuetze', 'shade', 'smile', 'swalk', 't',
                                 'tencel', 'tinte', 'transparent', 'trench', 'type', 'zenga', 'zinn']

    new_color_list_red    = ["red"]    * len(old_color_list_red)
    new_color_list_pink   = ["pink"]   * len(old_color_list_pink)
    new_color_list_violet = ["violet"] * len(old_color_list_violet)
    new_color_list_blue   = ["blue"]   * len(old_color_list_blue)
    new_color_list_cyan   = ["cyan"]   * len(old_color_list_cyan)
    new_color_list_green  = ["green"]  * len(old_color_list_green)
    new_color_list_yellow = ["yellow"] * len(old_color_list_yellow)
    new_color_list_orange = ["orange"] * len(old_color_list_orange)
    new_color_list_brown  = ["brown"]  * len(old_color_list_brown)
    new_color_list_white  = ["white"]  * len(old_color_list_white)
    new_color_list_grey   = ["grey"]   * len(old_color_list_grey)
    new_color_list_black  = ["black"]  * len(old_color_list_black)


    new_color_list_beige    = ["beige"]   * len(old_color_list_beige)
    new_color_list_skintone = ["skintone"] * len(old_color_list_skintone)

    new_color_list_multi = ["multi"] * len(old_color_list_multi)
    new_color_list_not_sorted = ["none"] * len(old_color_list_not_sorted)

    old_color_list = old_color_list_red + old_color_list_pink + old_color_list_violet + \
                     old_color_list_blue  + old_color_list_cyan + old_color_list_green + \
                     old_color_list_yellow + old_color_list_orange + old_color_list_brown + \
                     old_color_list_white + old_color_list_grey + old_color_list_black + \
                     old_color_list_beige + old_color_list_skintone + \
                     old_color_list_multi + old_color_list_not_sorted

    new_color_list = new_color_list_red + new_color_list_pink + new_color_list_violet + \
                     new_color_list_blue + new_color_list_cyan + new_color_list_green + \
                     new_color_list_yellow + new_color_list_orange + new_color_list_brown + \
                     new_color_list_white + new_color_list_grey + new_color_list_black + \
                     new_color_list_beige + new_color_list_skintone + \
                     new_color_list_multi + new_color_list_not_sorted

    if colour_tag != "none":
        for base_old_color, base_new_color in zip(old_color_list, new_color_list):
            pattern = base_old_color
            searchObj = re.search(pattern, colour_tag, re.M)
            if searchObj is not None:
                colour_tag_out = base_new_color
                #print "we found this word <%s> reduced to <%s>." % (searchObj.group(0),colour_tag_out)
                break

    return colour_tag_out



def fun_detect_style(text_data):
    """
    The function uses regex to check if the subcategories as we have defined then are present in the
    text data of an item.


    Args:
        text_data: a string (for example title + description)
    Returns:
        subcat_tag_out: a string (none is nothing is found)
    """

    list_base_style_en = ['elegant',
                           'chic',
                           'urban',
                           'casual',
                           'sporty',
                           'bohemian',
                           'classic',
                           'glam',
                           'sophisticated',
                           'traditional',
                           'preppy',
                           'street',
                           'punk',
                           'sexy',
                           'tomboy',
                           'goth',
                           'romantic',
                           'hipster',
                           'vibrant']

    list_base_style_de = ['elegant',
                            'chic',
                            'urban',
                            'lässig',
                            'sportlich',
                            'bohemian',
                            'klassisch',
                            'glamourös',
                            'anspruchsvoll',
                            'traditionell',
                            'preppy',
                            'street',
                            'rockig',
                            'attraktiv',
                            'tomboy',
                            'goth',
                            'romantisch',
                            'hipster',
                            'vibrant']

    #
    # --> where to we put Zeitlos???
    #

    text_data = text_data.lower()

    style_tag_out = "none"

    if type(text_data) != str:  # or type(text_gender) != "str" or type(text_cat) != "str":
        style_tag_out = "none"
    else:
        # for base_color_en, base_color_de in zip(list_base_colors_de, list_base_colors_en):
        for (base_style_en, base_style_de) in zip(list_base_style_en, list_base_style_de):
            pattern = base_style_en + '|' + base_style_de
            searchObj = re.search(pattern, text_data, re.M)
            if searchObj is not None:
                style_tag_out = base_style_en
                break

    return style_tag_out


def fun_detect_subcategory(text_gender, text_cat, text_data):
    """
    The function uses regex to check if the subcategories as we have defined them are present in the
    text data of an item.

    It checks first the category.


    Args:
        text_gender: a string "male" or "female"
        text_cat: a string (the reference base category or "new_category" in the csv file
        text_data: a string (for example title + description)
    Returns:
        subcat_tag_out: a string (none is nothing is found)
    """

    subcat_tag_out = "none"

    if type(text_data) != str:# or type(text_gender) != "str" or type(text_cat) != "str":
        subcat_tag_out = "none"
    else:
        if text_gender == "male":
            #print "we have a male clothes."
            list_base_tshirt_en = ["crew neck", "crew neck", "v neck", "polo shirt", "tank top"]
            list_base_tshirt_de = ["Rundhals", r"\bRundhalsausschnitt\b", r"\bV-Ausschnitt\b", r"\bPolo shirt\b",
                                   r"\bTank Top"]

            list_base_trouser_en = ["chinos", "cargo pants", "sweatpants", "jeans", "corduroys", "dress pants",
                                    "linen trousers",
                                    "leather pants"]
            list_base_trouser_de = ["Chinos", "Cargohose", "Jogginghose", "Jeans", "Cordhose", "Anzughose", "Leinenhose",
                                    "Lederhose"]

            list_base_jacket_en = ["jacket", "blouson", "bomber jacket", "down jacket", "functional jacket", "denim jacket",
                                   "parka", "quilting jacket", "windbreaker", "track jacket", "winter jacket",
                                   "coat", "leather jacket", "imitation leather jacket", "fur jacket"]
            list_base_jacket_de = ["Jacke", "Blouson", "Bomberjacke", "Daunenjacke", "Funktionsjacke", "Jeansjacke",
                                   "Parka",
                                   "Steppjacke", "Windbreaker", "Trainingsjacke", "Winterjacke", "Mantel", "Lederjacke",
                                   "Kunstlederjacke", "Felljacke"]

            list_base_pullover_en = ["hoodie", "turtleneck", "knitted sweater", "v neck", "sweatshirt"]
            list_base_pullover_de = ["Kapuzenpullover", "Rollkragenpullover", "Strickpullover", "V-Auschnitt", "Sweatshirt"]

            list_base_shirt_en = ["business shirt", "leisure shirt", "hawaiian", "denim shirt"]
            list_base_shirt_de = ["Businesshemd", "Freizeithemd", "Hawaiihemd", "Jeanshemd"]

            list_base_tie_en = ["wide tie", "bow tie", "narrow tie"]
            list_base_tie_de = ["Breite Krawatte", "Fliege", "Schmale Krawatte"]

            list_base_socks_en = ["footsies", "knee socks", "short socks", "sport socks"]
            list_base_socks_de = ["Füßlinge", "Kniestrümpfe", "Kurzsocken", "Sportsocken"]

            list_base_hat_en = ["knit cap", "cap", "hat"]
            list_base_hat_de = ["Strickmütze", "Kappe", "Hut"]

            list_base_glasses_en = ["frameless", "half frame", "full frame"]
            list_base_glasses_de = ["Randlos", "Vollrand", "Halbrand"]

            list_base_shorts_en = ["jean shorts", "shorts", "sport shorts", "trekking shorts"]
            list_base_shorts_de = ["Jeansshorts", "Shorts", "Sportshorts", "Trekkingshorts"]

            old_list = list_base_tshirt_en + list_base_tshirt_de + \
                       list_base_trouser_en + list_base_trouser_de + \
                       list_base_jacket_en + list_base_jacket_de + \
                       list_base_pullover_en + list_base_pullover_de + \
                       list_base_shirt_en + list_base_shirt_de + \
                       list_base_tie_en + list_base_tie_de + \
                       list_base_socks_en + list_base_socks_de + \
                       list_base_tie_en + list_base_hat_de + \
                       list_base_glasses_en + list_base_glasses_de + \
                       list_base_shorts_en + list_base_shorts_de

            new_list = list_base_tshirt_en + list_base_tshirt_en + \
                       list_base_trouser_en + list_base_trouser_en + \
                       list_base_jacket_en + list_base_jacket_en + \
                       list_base_pullover_en + list_base_pullover_en + \
                       list_base_shirt_en + list_base_shirt_en + \
                       list_base_tie_en + list_base_tie_en + \
                       list_base_socks_en + list_base_socks_en + \
                       list_base_tie_en + list_base_hat_en + \
                       list_base_glasses_en + list_base_glasses_en + \
                       list_base_shorts_en + list_base_shorts_en

            for base_cat in ["tshirt","trousers","jacket","pullover","shirt","tie","socks","hat","glasses","shorts"]:
                #print "Subcat in search for a <%s> <%s> clothe." % (base_cat, text_gender)
                if text_cat == base_cat:
                    #print "This is category: %s." % base_cat
                    for (base_old, base_new) in zip(old_list, new_list):
                        pattern = base_old+'|'+base_new
                        searchObj = re.search(pattern, text_data, re.M)

                        if searchObj is not None:
                            subcat_tag_out = base_new
                            #print "we found this word <%s> reduced to <%s>." % (searchObj.group(0), subcat_tag_out)
                            break # --> we leave the search

                    break # --> it wasn't in any of the category

        elif text_gender == "female":
            #print "we have a female clothes."
            list_base_trouser_en = ["7/8 pants","dress pants","bootcut trousers","flared trousers","pleated pants",
                    "bow fold pants","capri pants","cargo pants","chino pants","culottes","harem pants","dungarees",
                    "leather pants","leggings","sport pants","sweat pants"]
            list_base_trouser_de = ["7/8 Hose","Anzughose","Bootcut-Hose","Schlaghose","Bundfaltenhose",
                    "Bügelfaltenhose","Caprihose","Cargohose","Chinohose","Culottes","Haremshose","Latzhose",
                    "Lederhose","Leggings","Sporthose","Sweathose"]

            list_base_jacket_en = ["biker jacket","blouson","bolero","bomber jacket","down jacket","fur jacket",
                       "denim jacket","short jacket","leather jacket","parka","coat","track jacket","windbreaker","winter jacket"]
            list_base_jacket_de = ["Bikerjacke","Blousonjacke","Boleros","Bomberjacke","Daunenjacke","Felljacke",
                    "Jeansjacke","Kurzjacke","Lederjacke","Parka","Mantel","Trainingsjacke","Windbreaker","Winterjacke"]

            list_base_pullover_en = ["hoodie","turtle neck pullover","sleeveless pullover","crew neck","v-neck"]
            list_base_pullover_de = ["Kapuzenpullover","Rollkragenpullover","Pullunder","Rundkragen","V-Ausschnitt"]

            list_base_tshirt_en = ["v-neck","crew-neck","polo shirt","tank top","turtle neck shirt"]
            list_base_tshirt_de = ["V-Auschnitt","Rundhals","Polo Shirt","Tank Top","Rollkragenshirt"]

            list_base_blouse_en = ["blouse","shirt blouse","tunic"]
            list_base_blouse_de = ["Bluse","Hemdblouse","Tunika"]

            list_base_dress_en = ["summer dress","evening dress","blouses dress","jersey dress","shift dress","jean dress",
                "knit dress"]
            list_base_dress_de = ["Sommerkleid","Abendkleid","Blusenkleid","Jerseykleid","Etuikleid","Jeanskleid",
                "Strickkleid"]

            list_base_skirt_en = ["mini skirt","denim skirt","pleated skirt","A- line skirt","pencil skirt","leather skirt",
                "balloon skirt","maxi skirt"]
            list_base_skirt_de = ["Minirock","Jeansrock","Faltenrock","A-Linienrock","Bleistiftrock","Lederrock"
                "Ballonrock","Maxirock"]

            list_base_socks_en = ["footsies","knee socks","short socks","sport socks","overknee"]
            list_base_socks_de = ["Füßlinge","Kniestrümpfe","Kurzsocken","Sportsocken","overknee"]

            list_base_swimwear_en = ["bikini","swimsuit"]
            list_base_swimwear_de = ["Bikini","Badeanzug"]

            list_base_shorts_en = ["jean shorts","shorts","sport shorts","trekking shorts"]
            list_base_shorts_de = ["Jeansshorts","Shorts","Sportshorts","Trekkingshorts"]

            list_base_glasses_en = ["frameless","half frame", "full frame"]
            list_base_glasses_de = ["Randlos","Vollrand","Halbrand"]

            old_list = list_base_trouser_en + list_base_trouser_de + \
                       list_base_jacket_en + list_base_jacket_de + \
                       list_base_pullover_en + list_base_pullover_de + \
                       list_base_tshirt_en + list_base_tshirt_de + \
                       list_base_blouse_en + list_base_blouse_de + \
                       list_base_dress_en + list_base_dress_de + \
                       list_base_skirt_en + list_base_skirt_de + \
                       list_base_socks_en + list_base_socks_de + \
                       list_base_swimwear_en + list_base_swimwear_de + \
                       list_base_shorts_en + list_base_shorts_de + \
                       list_base_glasses_en + list_base_glasses_de

            new_list = list_base_trouser_en + list_base_trouser_en + \
                       list_base_jacket_en + list_base_jacket_en + \
                       list_base_pullover_en + list_base_pullover_en + \
                       list_base_tshirt_en + list_base_tshirt_en + \
                       list_base_blouse_en + list_base_blouse_en + \
                       list_base_dress_en + list_base_dress_en + \
                       list_base_skirt_en + list_base_skirt_en + \
                       list_base_socks_en + list_base_socks_en + \
                       list_base_swimwear_en + list_base_swimwear_en + \
                       list_base_shorts_en + list_base_shorts_en + \
                       list_base_glasses_en + list_base_glasses_en

            #print "The subcat for %s is not done yet, come back later." % text_gender
            for base_cat in ["trouser", "jacket", "pullover", "tshirt", "blouse", "dress", "skirt", "socks",
                             "swimwear","shorts","glasses"]:
                # print "Subcat in search for a <%s> <%s> clothe." % (base_cat, text_gender)
                if text_cat == base_cat:
                    # print "This is category: %s." % base_cat
                    for (base_old, base_new) in zip(old_list, new_list):
                        pattern = base_old + '|' + base_new
                        searchObj = re.search(pattern, text_data, re.M)

                        if searchObj is not None:
                            subcat_tag_out = base_new
                            # print "we found this word <%s> reduced to <%s>." % (searchObj.group(0), subcat_tag_out)
                            break  # --> we leave the search

                    break  # --> it wasn't in any of the category

    return subcat_tag_out

def fun_detect_subcategory_woman(text_data):
    """
    The function uses regex to check if the subcategories as we have defined then are present in the
    text data of an item.


    Args:
        text_data: a string (for example title + description)
    Returns:
        subcat_tag_out: a string (none is nothing is found)
    """

    list_base_subcat_en = ['elegant',
                           'chic',
                           'urban',
                           'casual',
                           'sporty',
                           'bohemian',
                           'classic',
                           'glam',
                           'sophisticated',
                           'traditional',
                           'preppy',
                           'street',
                           'punk',
                           'sexy',
                           'tomboy',
                           'goth',
                           'romantic',
                           'hipster',
                           'vibrant']

    list_base_subcat_de = ['elegant',
                            'chic',
                            'urban',
                            'lässig',
                            'sportlich',
                            'bohemian',
                            'klassisch',
                            'glamourös',
                            'anspruchsvoll',
                            'traditionell',
                            'preppy',
                            'street',
                            'rockig',
                            'attraktiv',
                            'tomboy',
                            'goth',
                            'romantisch',
                            'hipster',
                            'vibrant']

    subcat_tag_out = "none"

    # for base_color_en, base_color_de in zip(list_base_colors_de, list_base_colors_en):
    for (base_subcat_en, base_subcat_de) in zip(list_base_subcat_en, list_base_subcat_de):
        pattern = text_data
        # the "search" is better than "match" as it check all over the string and not
        # only the beginning of the string in which we search the pattern.
        searchObj = re.search(pattern, text_data, re.M)
        if searchObj is not None:
            subcat_tag_out = base_subcat_en
            break

    return subcat_tag_out


def fun_extract_material_from_clothe(item_description):
    """
    The function extracts the material of an item when it is possible. Following the same approach as for the
    color reduction, it uses regex to find out it words describing material are present in the item_description.
    Args:
        item_description: (str)

    Returns:
        item_material: (str)
    """

    item_material = "unknown textile"

    #Schurwolle | new wool

    # denim

    # lt aged / beached --> jeans

    # eye allove -> pattern as moon dot
    #wolle | wool


    #Seile | silk

    # regex for item + item + mix in case of mix textile
    # word -> Materialmix

    # regex for item [und] item Material

    return item_material
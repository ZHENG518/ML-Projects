from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import numpy as np

# nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords = stopwords.words('english')

import gensim
from gensim import corpora
import logging

# for gensim to output some progress information while it's training
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def clean_doc(doc):
    # 去除停用词，统一为小写
    words = []
    for word in doc:
        word = word.lower()
        if word not in stopwords:
            words.append(word)
    return words

def bag_of_words(train_docs, test_docs):
    '''
    文本bow向量化
    :param train_docs: 训练样本分词后的文档列表，每个文档为一个words list
    :param test_docs: 测试样本分词后的文档列表，每个文档为一个words list
    :return: 文档的向量表示
    '''
    train_bows = [dict(Counter(clean_doc(doc))) for doc in train_docs]
    test_bows = [dict(Counter(clean_doc(doc))) for doc in test_docs]

    vectorizer = DictVectorizer(sparse=False)
    train_vecs = vectorizer.fit_transform(train_bows)
    test_vecs = vectorizer.transform(test_bows)

    print('train_shape:',train_vecs.shape)
    print('test_shape:',test_vecs.shape)
    return train_vecs, test_vecs

def tf_idf(train_docs, test_docs):
    '''
    文本tf_idf向量化
    :param train_docs: 训练样本分词后的文档列表，每个文档为一个words list
    :param test_docs: 测试样本分词后的文档列表，每个文档为一个words list
    :return: 文档的向量表示
    '''
    train_docs = [' '.join(clean_doc(doc)) for doc in train_docs]
    test_docs = [' '.join(clean_doc(doc)) for doc in test_docs]

    vectorizer = TfidfVectorizer()
    train_vecs = vectorizer.fit_transform(train_docs)
    test_vecs = vectorizer.transform(test_docs)

    print('train_shape:',train_vecs.shape)
    print('test_shape:',test_vecs.shape)
    return train_vecs, test_vecs

def preprocessing(docs):
    # we filter stopwords using nltk stopword list
    text_data = [[word.lower() for word in doc if (len(word) > 4 and word.lower() not in stopwords)] for doc in docs]
    # Dictionary encapsulates the mapping between normalized words and their integer ids.
    dictionary = corpora.Dictionary(text_data)
    # no_below: Keep tokens which are contained in at least no_below documents.
    # no_above: Keep tokens which are contained in no more than no_above documents
    #           (fraction of total corpus size, not an absolute number).
    dictionary.filter_extremes(no_below=10, no_above=0.5)
    # Filter out the 20 most frequent tokens that appear in the documents.
    dictionary.filter_n_most_frequent(20)
    # convert documents to BOW representations
    corpus = [dictionary.doc2bow(doc) for doc in text_data]

    return corpus, dictionary

def lda_vectorizer(train_docs, test_docs, num_topics = 50):
    preprocessed_train, train_dictionary = preprocessing(train_docs)
    preprocessed_test, test_dictionary = preprocessing(test_docs)
    model = gensim.models.LdaModel(preprocessed_train, id2word=train_dictionary,
                                   num_topics=num_topics, alpha='auto', eta='auto',passes=10)
    train_vectors = [model[doc] for doc in preprocessed_train]
    test_vectors = [model[doc] for doc in preprocessed_test]

    def to_matrix(vectors):
        matrix = np.zeros((len(vectors), num_topics))
        for i in range(len(vectors)):
            for t in vectors[i]:
                matrix[i][t[0]] = t[1]
        return matrix

    train_matrix = to_matrix(train_vectors)
    test_matrix = to_matrix(test_vectors)

    print(train_matrix.shape)
    print(test_matrix.shape)
    return train_matrix, test_matrix

if __name__ == '__main__':
    from sklearn.svm import LinearSVC
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import accuracy_score, classification_report
    from src.util import load_data

    train_text, train_label, test_text, test_label = load_data()
    # train_x, test_x = bag_of_words(train_text, test_text)
    # train_x, test_x = tf_idf(train_text, test_text)
    train_x, test_x = lda_vectorizer(train_text, test_text)

    svm_clf = LinearSVC()
    train_pred = cross_val_predict(svm_clf, train_x, train_label, cv=10)
    print("train accuracy")
    print(accuracy_score(train_label, train_pred))
    print(classification_report(train_label, train_pred))

    svm_clf.fit(train_x, train_label)
    test_pred = svm_clf.predict(test_x)
    print("train accuracy")
    print(accuracy_score(test_label, test_pred))
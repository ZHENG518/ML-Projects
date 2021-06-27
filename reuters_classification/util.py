import nltk
# nltk.download('reuters')
from nltk.corpus import reuters

def load_data():
    training_set = []
    training_classifications = []

    test_set = []
    test_classifications = []

    topic = "acq"

    for file_id in reuters.fileids():
        if file_id.startswith("train"):
            training_set.append(reuters.words(file_id))
            if topic in reuters.categories(file_id):
                training_classifications.append(topic)
            else:
                training_classifications.append("not " + topic)
        else:
            test_set.append(reuters.words(file_id))
            if topic in reuters.categories(file_id):
                test_classifications.append(topic)
            else:
                test_classifications.append("not " + topic)

    return training_set, training_classifications, test_set, test_classifications

if __name__ == '__main__':
    from utils.plotting import histogram

    train_text, train_label, test_text, test_label = load_data()
    print('Train Size:', len(train_text))
    print('Test Size:', len(test_text))

    # 统计文本长度
    len_list = [len(doc) for doc in train_text+test_text]
    print('Longest:', max(len_list))
    print('Shortest:', min(len_list))
    print('Average Length:',round(sum(len_list)/len(len_list), 2))
    histogram([len for len in len_list if len <=550])

    print('hhh')
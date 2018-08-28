import numpy as np

class data_loader():
    def __init__(self, data_path = "./data/data.txt"):
        self.data_path = data_path
        self.tokenized_sentences = []
        self.word_dic = set()
        self.word_cnt = 0
        self.word_to_ix = {}
        self.ix_to_word = {}

        self.make_dict()

    def make_dict(self):
        with open("./data/data.txt", 'r') as f:
            for line in f.readlines():
                word_split = line.split()
                self.tokenized_sentences.append(word_split)
                for word in word_split:
                    self.word_dic.add(word)
        self.word_cnt = len(self.word_dic)
        self.ix_to_word, self.word_to_ix = {i: word for i, word in enumerate(self.word_dic)}, {word: i for i, word in enumerate(self.word_dic)}
        print("The number of words :", self.word_cnt)

    def transform(self, sentences, print = True):
        if sentences.ndim == 1:
            result = [self.ix_to_word[x] for x in sentences]
        else:
            result = [[self.ix_to_word[x] for x in sent] for sent in sentences]

        if print:
            return ' '.join(result).replace('<START_TOKEN> ','').replace(' <PAD>','').replace(' <END_TOKEN>','')
        else:
            return result

    def data_load(self):
        X_train = np.asarray([[self.word_to_ix[w] for w in sent[:-1]] for sent in self.tokenized_sentences])
        Y_train = np.asarray([[self.word_to_ix[w] for w in sent[1:]] for sent in self.tokenized_sentences])

        return X_train, Y_train


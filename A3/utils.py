import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


DOCSTART_LITERAL = '-DOCSTART-'


class CorpusReader:
    NEGATIVE_TABLE_SIZE = 1e8

    def __init__(self, inputFileName, min_count:int = 5, lang="zh") :
        self.negatives = []
        self.discards = []
        self.negpos = 0

        self.word2id = dict()
        self.id2word = dict()
        self.token_count = 0
        self.word_frequency = dict()

        self.lang = lang
        self.inputFileName = inputFileName
        self.read_words(min_count)
        # self.initTableNegatives()
        # self.initTableDiscards()

    def read_words(self, min_count):
        word_frequency = dict()
        for line in open(self.inputFileName, encoding="utf8"):
            if self.lang == "zh":
                words = list(line.strip())
            else:
                words = line.split()
            if len(words) > 0:
                for word in words:
                    self.token_count += 1
                    word_frequency[word] = word_frequency.get(word, 0) + 1
                    if self.token_count % 1000000 == 0:
                        print("Read " + str(int(self.token_count / 1000000)) + "M words.")

        wid = 0
        for w, c in sorted(word_frequency.items(), key=lambda x: x[1], reverse=True):
            if c < min_count: # filter out low frequency words
                continue
            self.word2id[w] = wid
            self.id2word[wid] = w
            self.word_frequency[wid] = c
            wid += 1
        print("Total vocabulary: " + str(len(self.word2id)))

    def initTableDiscards(self):
        t = 0.0001
        f = np.array(list(self.word_frequency.values())) / self.token_count
        self.discards = np.sqrt(t / f) + (t / f)

    def initTableNegatives(self):
        pow_frequency = np.array(list(self.word_frequency.values())) ** (3/4)
        words_pow = sum(pow_frequency)
        ratio = pow_frequency / words_pow
        count = np.round(ratio * CorpusReader.NEGATIVE_TABLE_SIZE)
        for wid, c in enumerate(count):
            self.negatives += [wid] * int(c)
        self.negatives = np.array(self.negatives)
        np.random.shuffle(self.negatives)

    def getNegatives(self, target, size):
        while True:
            response = self.negatives[self.negpos:self.negpos + size]
            self.negpos = (self.negpos + size) % len(self.negatives)
            if len(response) != size:
                response = np.concatenate((response, self.negatives[0:self.negpos]))
            if target in response: # prevent target word itself from being negative sample
                continue
            return response


def read_ner_data_from_connl(path_to_file):
    words = []
    tags = []

    with open(path_to_file, 'r', encoding='utf-8') as file:
        for line in file:
            splitted = line.split()
            if len(splitted) == 0:
                continue
            word = splitted[0]
            if word == DOCSTART_LITERAL:
                continue
            entity = splitted[-1]
            words.append(word)
            tags.append(entity)
        return words, tags


def get_batched(words, labels, size):
    for i in range(0, len(labels), size):
        yield (words[i:i + size], labels[i:i + size])


def load_embedding_dict(vec_path):
    embeddings_index = dict()
    with open(vec_path, 'r', encoding='UTF-8') as file:
        for line in tqdm(file.readlines()):
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    return embeddings_index


def get_tag_indices_from_scores(scores: np.ndarray):
    predicted = []
    for i in range(scores.shape[0]):
        predicted.append(int(np.argmax(scores[i])))
    return predicted


def build_training_visualization(model_name, train_metrics, losses, validation_metrics, path_to_save=None):
    figure = plt.figure(figsize=(20, 30))
    figure.suptitle(f'Visualizations of {model_name} training progress', fontsize=16)

    ax1 = figure.add_subplot(3, 1, 1)
    ax1.plot(losses)
    ax1.set_title("Loss through epochs")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")

    ax2 = figure.add_subplot(3, 1, 2)
    for metric, results in validation_metrics.items():
        ax2.plot(results, label=metric)
    ax2.legend(loc='upper left')
    ax2.set_title("Metrics through epochs")
    ax2.set_xlabel("Epochs")

    ax3 = figure.add_subplot(3, 1, 3)
    for metric, results in train_metrics.items():
        ax3.plot(results, label=metric)
    ax3.legend(loc='upper left')
    ax3.set_title("Results on dev set through epochs")
    ax3.set_xlabel("Epochs")

    if path_to_save:
        figure.savefig(path_to_save)
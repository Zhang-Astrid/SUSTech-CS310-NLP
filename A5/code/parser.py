import sys
import numpy as np
import torch
import argparse

from model import BaseModel, WordPOSModel
from parse_utils import DependencyArc, DependencyTree, State, parse_conll_relation
from get_train_data import FeatureExtractor

argparser = argparse.ArgumentParser()
argparser.add_argument('--model', type=str, default='model.pt')
argparser.add_argument('--words_vocab', default='words_vocab.txt')
argparser.add_argument('--pos_vocab', default='pos_vocab.txt')
argparser.add_argument('--rel_vocab', default='rel_vocab.txt')


class Parser(object):
    def __init__(self, extractor: FeatureExtractor, model_file: str, ):
        ### START YOUR CODE ###
        # TODO: Initialize the model
        word_vocab_size = len(extractor.word_vocab)
        pos_vocab_size = len(extractor.pos_vocab)
        output_size = len(extractor.rel_vocab)
        # self.model = BaseModel(word_vocab_size, output_size)  # 使用适当的词汇表大小和输出类别数量
        self.model = WordPOSModel(word_vocab_size, pos_vocab_size, output_size)
        ### END YOUR CODE ###
        self.model.load_state_dict(torch.load(model_file))
        self.model.eval()
        self.extractor = extractor

        # The following dictionary from indices to output actions will be useful
        self.output_labels = dict(
            [(index, action) for (action, index) in extractor.rel_vocab.items()]
        )

    def parse_sentence(self, words, pos):
        state = State(range(1, len(words)))
        state.stack.append(0)

        while state.buffer:
            ### START YOUR CODE ###
            # TODO: Extract the current state representation and make a prediction
            # Call self.extractor.get_input_repr_*()
            # current_state = self.extractor.get_input_repr_word(words, state)
            current_state = self.extractor.get_input_repr_wordpos(words, pos, state)  # 获取特征表示
            with torch.no_grad():
                input_tensor = torch.from_numpy(np.array([current_state], dtype=np.int64))  # 转为tensor
                prediction = self.model(input_tensor)  # 使用模型进行预测

            best_action_idx = prediction.argmax(dim=1).item()
            best_action = self.output_labels[best_action_idx]
            action_type = best_action[0]

            if action_type == "shift":
                state.shift()
            elif action_type == "left_arc" and len(state.stack) > 0:
                state.left_arc(best_action[1])
            elif action_type == "right_arc" and len(state.stack) > 0:
                state.right_arc(best_action[1])
            else:
                state.shift()

        ### START YOUR CODE ###
        # TODO: Go through each relation in state.deps and add it to the tree by calling tree.add_deprel()
        tree = DependencyTree()
        for dep in state.deps:
            # 将每个依赖关系添加到树中
            tree.add_deprel(DependencyArc(dep[1], words[dep[1]], pos[dep[1]], dep[0], dep[2]))  # dep[0] - 1?
        if not tree.deprels:
            # print('Warning: No deprels found, replaced with DependencyArc(0, "<ROOT>", "<ROOT>", 0, "root")')
            tree.add_deprel(DependencyArc(0, "<ROOT>", "<ROOT>", 0, "root"))
        ### END YOUR CODE ###

        return tree


if __name__ == "__main__":
    args = argparser.parse_args()
    try:
        word_vocab_file = open(args.words_vocab, "r")
        pos_vocab_file = open(args.pos_vocab, "r")
        rel_vocab_file = open(args.rel_vocab, "r")
    except FileNotFoundError:
        print(f'Could not find vocabulary files {args.words_vocab}, {args.pos_vocab}, and {args.rel_vocab}')
        sys.exit(1)

    extractor = FeatureExtractor(word_vocab_file, pos_vocab_file, rel_vocab_file)
    parser = Parser(extractor, args.model)

    # Test an example sentence, 3rd example from dev.conll
    words = [None, 'The', 'bill', 'intends', 'to', 'restrict', 'the', 'RTC', 'to', 'Treasury', 'borrowings', 'only',
             ',', 'unless', 'the', 'agency', 'receives', 'specific', 'congressional', 'authorization', '.']
    pos = [None, 'DT', 'NN', 'VBZ', 'TO', 'VB', 'DT', 'NNP', 'TO', 'NNP', 'NNS', 'RB', ',', 'IN', 'DT', 'NN', 'VBZ',
           'JJ', 'JJ', 'NN', '.']

    tree = parser.parse_sentence(words, pos)
    print(tree)

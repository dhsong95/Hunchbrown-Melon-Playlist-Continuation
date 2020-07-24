#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
세종 코퍼스와 khaiii 학습 코퍼스를 원하는 형태로 변환하는 스크립트.
- 입력: sejong: 세종 코퍼스, train: khaiii 학습 코퍼스
- 출력: raw: 원문, khaiii: khaiii 바이너리 프로그램의 출력 포맷
__author__ = 'Jamie (jamie.lim@kakaocorp.com)'
__copyright__ = 'Copyright (C) 2019-, Kakao Corp. All rights reserved.'
"""


###########
# imports #
###########
from argparse import ArgumentParser, Namespace
import logging
import random
import sys
from typing import List

from khaiii.munjong.sejong_corpus import sents
from khaiii.resource.resource import Resource
from khaiii.train.dataset import PosDataset


#########
# types #
#########
class Sentence:
    """
    sentence object
    """
    def __init__(self):
        self.words = []
        self.morphs = []

    def merge_words(self, rate: float = 0.0):
        """
        어절을 무작위로 합친다.
        Args:
            rate:  비율
        """
        if rate <= 0.0:
            return
        idx = 0
        while idx < len(self.words)-1:
            if random.random() >= rate:
                idx += 1
                continue
            self.words[idx] += self.words[idx+1]
            self.morphs[idx] += ' + ' + self.morphs[idx+1]
            del self.words[idx+1]
            del self.morphs[idx+1]

    def __str__(self):
        words_str = [f'{w}\t{m}' for w, m in zip(self.words, self.morphs)]
        return '\n'.join(words_str) + '\n'

    def raw(self):
        """
        raw text
        Returns:
            raw text in sentence
        """
        return ' '.join(self.words)

    @classmethod
    def load_sejong(cls) -> List['Sentence']:
        """
        load from Sejong corpus
        Returns:
            list of sentences
        """
        sentences = []
        for sent in sents(sys.stdin):
            sentence = Sentence()
            for word in sent.words:
                sentence.words.append(word.raw)
                sentence.morphs.append(' + '.join([str(m) for m in word.morphs]))
            sentences.append(sentence)
        return sentences

    @classmethod
    def load_train(cls, rsc_src: str) -> List['Sentence']:
        """
        load from khaiii training set
        Returns:
            list of sentences
        """
        restore_dic = Resource.load_restore_dic(f'{rsc_src}/restore.dic')
        sentences = []
        for sent in PosDataset(None, restore_dic, sys.stdin):
            sentence = Sentence()
            for word in sent.pos_tagged_words:
                sentence.words.append(word.raw)
                sentence.morphs.append(' + '.join([str(m) for m in word.pos_tagged_morphs]))
            sentences.append(sentence)
        return sentences


#############
# functions #
#############
def run(args: Namespace):
    """
    run function which is the start point of program
    Args:
        args:  program arguments
    """
    random.seed(args.seed)

    sentences = []
    if args.input_format == 'sejong':
        sentences = Sentence.load_sejong()
    elif args.input_format == 'train':
        sentences = Sentence.load_train(args.rsc_src)
    else:
        raise ValueError(f'invalid input format: {args.input_format}')

    for sentence in sentences:
        sentence.merge_words(args.merge_rate)
        if args.output_format == 'raw':
            print(sentence.raw())
        elif args.output_format == 'khaiii':
            print(str(sentence))
        else:
            raise ValueError(f'invalid output format: {args.output_format}')


########
# main #
########
def main():
    """
    main function processes only argument parsing
    """
    parser = ArgumentParser(description='세종 코퍼스와 khaiii 학습 코퍼스를 원하는 형태로 변환하는 스크립트')
    parser.add_argument('-i', '--input-format', help='input format (sejong, train)', metavar='FMT',
                        required=True)
    parser.add_argument('-o', '--output-format', help='output format (raw, khaiii)', metavar='FMT',
                        required=True)
    parser.add_argument('--rsc-src', help='resource source dir <default: ../rsc/src>',
                        metavar='DIR', default='../rsc/src')
    parser.add_argument('--input', help='input file <default: stdin>', metavar='FILE', )
    parser.add_argument('--output', help='output file <default: stdout>', metavar='FILE')
    parser.add_argument('--seed', help='random seed <default: 1234>', metavar='NUM', type=int,
                        default=1234)
    parser.add_argument('--merge-rate', help='word merge rate', metavar='REAL', type=float,
                        default=0.0)
    parser.add_argument('--debug', help='enable debug', action='store_true')
    args = parser.parse_args()

    if args.input:
        sys.stdin = open(args.input, 'r', encoding='UTF-8')
    if args.output:
        sys.stdout = open(args.output, 'w', encoding='UTF-8')
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    run(args)


if __name__ == '__main__':
    main()

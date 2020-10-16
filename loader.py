#encoding:utf-8
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import  to_categorical
import codecs
import sentencepiece as spm
import re
import numpy as np
import jieba


sp = spm.SentencePieceProcessor()
sp.Load("./model_dir/newspiece_8000.model")
SPIECE_UNDERLINE = '▁'

re_han = re.compile(u"([\u4E00-\u9FD5a-zA-Z]+)")

def read_file(filename):
    """read data yield label, text"""
    with codecs.open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                line = line.rstrip()
                line=line.split('\t')
                assert len(line) == 2
                label, text= line
                yield label, text
            except:
                pass

def spm_ids(sp_model, text):
    """
    use sentencepiece model convert text to token
    """
    pieces = sp_model.EncodeAsPieces(text)
    pieces=[ p.replace(SPIECE_UNDERLINE, '') for p in pieces  ]
    pieces=[p for p in pieces if re_han.match(p)]
    ids = [sp_model.PieceToId(piece) for piece in pieces]
    return  ids

def jieba_ids(vocab_dir,text):
    """
    use jiab segment cut text and convert word to token
    """
    words=codecs.open(vocab_dir,'r',encoding='utf-8').read().strip().split('\n')
    word_to_id=dict(zip(words,range(len(words))))
    words = []
    blocks = re_han.split(text)
    for blk in blocks:
        if re_han.match(blk):
            for w in jieba.cut(blk):
                    words.append(w)
    ids=[word_to_id[x] for x in words if x in word_to_id]
    return ids



def convert_examples_to_tokens(input_dir,vocab_dir,max_length=200,spm=False):
    """
    process train data and return (x_pad,y_pad) for model
    """

    label_dict={'体育':0, '财经':1, '房产':2, '家居':3, '教育':4,
                '科技':5, '时尚':6, '时政':7, '游戏':8, '娱乐':9}

    data_id,label_id=[],[]
    for label,text in read_file(input_dir):
        if spm:
            data_id.append(spm_ids(sp, text))
        else:
            data_id.append(jieba_ids(vocab_dir, text))

        label_id.append(label_dict[label])

    x_pad=pad_sequences(data_id,max_length,padding='post', truncating='post')
    y_pad=to_categorical(label_id)
    return x_pad,y_pad



if __name__ == '__main__':
    text='新浪体育讯北京时间4月27日，NBA季后赛首轮洛杉矶湖人主场迎战新奥尔良黄蜂'
    print(text)
    print('---------------sentencepiece-----------')
    pieces=sp.EncodeAsPieces(text)
    pieces = [p.replace(SPIECE_UNDERLINE, '') for p in pieces]
    print('\\'.join(pieces))
    print('---------------jieba-----------')
    print('\\'.join(jieba.cut(text)))
    # print(spm_ids(sp, text))

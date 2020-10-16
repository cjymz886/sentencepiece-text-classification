import tensorflow as tf
from model import *
from loader import *
from sklearn import metrics
import numpy as np


def test():
    cfg = TextConfig()
    model = tf.keras.models.load_model('./model_dir/text_cnn.h5')
    test_x,test_y=convert_examples_to_tokens(cfg.test_dir,cfg.vocab_dir, max_length=cfg.seq_length,spm=True)
    res=model.evaluate(test_x,test_y, verbose=0)
    print(res)
    test_y= np.argmax(test_y,1)

    pred_y = np.argmax(model.predict(test_x, batch_size=32), 1)

    categories = ['体育', '财经', '房产', '家居', '教育' ,'科技', '时尚', '时政', '游戏', '娱乐']
    report = metrics.classification_report(test_y, pred_y, target_names=categories)
    print(report)

if __name__=="__main__":
    test()
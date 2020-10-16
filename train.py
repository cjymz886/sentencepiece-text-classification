from model import *
from loader import *
import os
from tensorflow.keras.callbacks import ReduceLROnPlateau
import numpy as np
import time
import pickle


def train():

    cfg=TextConfig()
    model=cnn_model(cfg)

    t1 = time.time()
    train_x,train_y=convert_examples_to_tokens(cfg.train_dir,cfg.vocab_dir,max_length=cfg.seq_length,spm=cfg.spm)
    print("Loading train data cost %.3f seconds...\n" % (time.time() - t1))

    val_x, val_y = convert_examples_to_tokens(cfg.val_dir,cfg.vocab_dir, max_length=cfg.seq_length,spm=cfg.spm)
    indices=np.random.permutation(np.arange(len(train_x)))
    train_x=train_x[indices]
    train_y=train_y[indices]


    reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.9, patience=1, min_lr=0.00001)
    history=model.fit(train_x,train_y,epochs=cfg.num_epochs,batch_size=cfg.batch_size,
                      verbose=1,validation_data=(val_x,val_y), callbacks=[reduce_lr],)
    model.save('./model_dir/text_cnn_spm.h5')

    with open('./model_dir/history.pickle', 'wb') as f:
        pickle.dump(history.history, f)


if __name__ == "__main__":
    train()

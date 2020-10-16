from tensorflow import keras
from tensorflow.keras import layers



class TextConfig():

    embedding_size=100    #dimension of word embedding
    vocab_size=8000       #number of vocabulary

    seq_length=600         #max length of sentence

    num_filters=128    #number of convolution kernel
    filter_sizes=[2,3,4]
    num_classes=10
    hidden_unit=128

    drop_prob=0.4          #droppout
    lr= 1e-3               #learning rate

    num_epochs=10          #epochs
    batch_size=64         #batch_size
    spm=True             #use sentencepiece

    train_dir='./data/cnews.train.txt'
    val_dir='./data/cnews.val.txt'
    test_dir='./data/cnews.test.txt'
    vocab_dir='./data/vocab.txt'




def cnn_model(cfg):
    def convolution():
        inn = layers.Input(shape=(cfg.seq_length, cfg.embedding_size, 1))
        cnns = []
        for size in cfg.filter_sizes:
            conv = layers.Conv2D(filters=cfg.num_filters, kernel_size=(size, cfg.embedding_size),
                                 strides=1, padding='valid', activation='relu',
                                 kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.1),
                               )(inn)
            pool = layers.MaxPool2D(pool_size=(cfg.seq_length - size + 1, 1), padding='valid')(conv)
            pool = layers.BatchNormalization()(pool)
            cnns.append(pool)
        outt = layers.concatenate(cnns)
        model = keras.Model(inputs=inn, outputs=outt)
        return model

    model = keras.Sequential([
        layers.Embedding(input_dim=cfg.vocab_size, output_dim=cfg.embedding_size,
                        input_length=cfg.seq_length),
        layers.Reshape((cfg.seq_length, cfg.embedding_size, 1)),
        convolution(),
        layers.Flatten(),
        layers.Dropout(cfg.drop_prob),
        layers.Dense(cfg.num_classes, activation='softmax', kernel_initializer='random_normal', bias_initializer='zero'
                     ,kernel_regularizer=keras.regularizers.l2(0.01))

    ])
    model.compile(optimizer=keras.optimizers.Adam(),
                 loss=keras.losses.CategoricalCrossentropy(),
                 metrics=['accuracy',
                          keras.metrics.Recall()])
    print(model.summary())
    return model


# Text classification with Sentencepiece and CNN
Sentencepiece是google开源的文本Tokenzier工具，其主要原理是利用统计算法，在语料库中生成一个类似分词器的工具，外加可以将词token化的功能；只是不同开源的分词器，它会频繁出现的字符串作为词，然后形成词库进行切分，所以它会切分的粒度更大。<br>
<br>
例如“机器学习领域“这个文本，按jieba会分“机器/学习/领域”，但你想要粒度更大的切分效果，如“机器学习/领域”或者不切分，这样更有利于模型捕捉更多N-gram特征。为实现这个，你可能想到把对应的大粒度词加到词表中就可以解决，但是添加这类词是很消耗人力。然而对于该问题，sentencepiece可以得到一定程度解决，甚至完美解决你的需求。<br>
<br>
基于上述的场景，本项目主要探究利用sentencepiece进行文本分类，对比开源的分词器，会有怎样的效果。在实验中，选择的是中文新闻文本数据集，对比的开源分词器选择的是jieba。若想对sentencepiece有更多的了解，可以查看[sentencepiece原理与实践](https://zhuanlan.zhihu.com/p/159200073)。<br>


1 环境
=
python3 <br>
tensorflow2.0 <br>
jieba <br>
sentencepiece <br>
numpy <br>

2 数据集及前期处理
=
本实验同样是使用THUCNews的一个子集进行训练与测试，数据集请自行到[THUCTC：一个高效的中文文本分类工具包](http://thuctc.thunlp.org/)下载，请遵循数据提供方的开源协议;<br><br>
文本类别涉及10个类别：categories = \['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']，每个分类6500条数据；<br><br>
cnews.train.txt: 训练集(5000*10)<br>
cnews.val.txt: 验证集(500*10)<br>
cnews.test.txt: 测试集(1000*10)<br><br>

其实，本项目是基于词级别的CNN for text classification, 只是这个词一个从jieba切分过来的，一个是sentencepiece训练的模型识别出来的。在预处理过程中，本项目中只是简单的过滤标点符号，数字类型的词，具体code体现在loader.py 文中的 re_han=re.compile(u"([\u4E00-\u9FD5a-zA-Z]+)")。<br><br>

3 超参数说明
=
~~~
class TextConfig():

    embedding_size=100    #dimension of word embedding
    vocab_size=8000     #number of vocabulary

    seq_length=600        #max length of sentence

    num_filters=128       #number of convolution kernel
    filter_sizes=[2,3,4]
    num_classes=10
    hidden_unit=128

    drop_prob=0.5          #droppout
    lr= 1e-3               #learning rate

    num_epochs=10          #epochs
    batch_size=64          #batch_size
    spm=True               #use sentencepiece

    train_dir='./data/cnews.train.txt'
    val_dir='./data/cnews.val.txt'
    test_dir='./data/cnews.test.txt'
    vocab_dir='./data/vocab.txt'
~~~
在与jieba对比的时，设定的vocab_size=8000,spm参数控制是否use sentencepiece，其他参数都是一致的；vocab.txt是用jieba切分后前8000的高频词；<br><br>

4 实验对比
=
**(1) 训练和验证准确率对比**
| 模型 | train_accuracy | val_accuracy | test_accuracy |
| ------| ------| ------| ------|
| jieba+cnn| 0.9988 |0.9686|0.9706|
|spm+cnn|0.9972 |0.9704|0.9667|
从训练结果来看，二者相差并不大，利用spm+cnn在验证集上有一定的提升,但在测试集jieba+cnn表现好一些。通过这些微小数据对比，个人是觉得利用sentencepiece相对jieba这类正规分词器来说，更容易过拟合些，另个角度来说，它捕捉的特征更多些，但也带来更多噪声特征的影响。<br><br>

**(2) 训练中损失变化对比**

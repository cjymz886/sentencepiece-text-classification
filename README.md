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
|jieba+word2vec+cnn|1.000|0.971|0.9723|
| jieba+cnn| 0.9988 |0.9686|0.9706|
|spm+cnn|0.9972 |0.9704|0.9667|

**(2) 训练中损失变化对比**
![image](https://github.com/cjymz886/sentencepiece-text-classification/blob/main/imgs/img_loss.png)

从训练结果来看，二者相差并不大，利用spm+cnn在验证集上有一定的提升,但在测试集jieba+cnn表现好一些。通过这些微小数据对比，个人觉得利用sentencepiece相对jieba这类正规分词器来说，更容易过拟合些，换个角度来说，它捕捉的特征更多些，但也带来更多噪声特征的影响。<br><br>

**(3) 载入数据集的消耗时间**
| 模型 | cost_time(seconds) |
| ------| ------|
| jieba+cnn| 475 |
|spm+cnn|80 |

对比jieba分词器，sentencepiece切分效率是它的近6倍，基于这个优势，是可以看出sentencepiece的使用价值的，尤其当处理的文档级的文本的时候。<br><br>

**(4) 是不是词表越大越好**<br><br>
在与jieba对比中，我选择的都是8000个高频词，可能有疑问：是不是词表越大效果会越好？对此，本实验在spm+cnn模型下，对比了词表8000,20000,320000(sentencepiece能训练的最大词表)，效果如下：
![image](https://github.com/cjymz886/sentencepiece-text-classification/blob/main/imgs/img_acc.png)

可以看出，随着词表增大，在验证集的表现越来越差。理论上不是词表越大越好吗，它毕竟降低了未登录词出现的概率。其实，我想是这样的，该新闻数据集的各个label区分度是很高的，也就是说影响每个label的特征都是很明显的，而且这些影响特征都是属于高频词汇的，如果加大词表，就相当于training过程中，让model学到很多label的噪声特征，导致在验证集上效果变小。<br><br>

还有一个原因：该数据集不论基于字，词，或者加上word2vec词向量，它的train_accuracy都很高，如果一个数据集的train_accuracy较低，增加词表应该会有正向的提升。<br><br>

**(4) spm不同词表下切分效果对比**<br><br>
在训练sentencepiece，可以设定输出词表的大小，本实验训练了8000,20000,320000三个级别的spm model，对比jieba，看看它们切分文本的效果，如下： <br><br>
| 模型 | text |
| ------| ------|
|no_segement|新浪体育讯北京时间4月27日，NBA季后赛首轮洛杉矶湖人主场迎战新奥尔良黄蜂|
|jieba|新浪\体育讯\北京\时间\4\月\27\日\，\NBA\季后赛\首轮\洛杉矶\湖人\主场\迎战\新奥尔良\黄蜂|
|spm_8000|\新浪体育讯北京时间\4\月\27\日\,\NBA\季后赛\首\轮\洛杉矶\湖人\主场\迎\战\新\奥\尔\良\黄蜂|
|spm_20000|\新浪体育讯北京时间\4\月\27\日\,\NBA\季后赛\首轮\洛杉矶湖人\主场迎战\新\奥尔良\黄蜂 |
|spm_320000|新浪体育讯北京时间\4\月\27\日\,\NBA\季后赛首轮\洛杉矶湖人主场迎战新奥尔良黄蜂 |

对比显示：随着词表增大，spm切分的粒度越来越大；三个spm模型都将“新浪体育讯北京时间”当一个词块，说明语料库中该词库出现的频率很高；在spm_320000模型下，将“洛杉矶湖人主场迎战新奥尔良黄蜂”切在一起，粒度是相当的大，直接是将一句话当成一个词块了。<br><br>

此外，可以看出spm_8000下粒度太细了，很多单字情况，这种情况明显是没有jieba效果好，也就影响了模型的训练，试想：如果在训练spm模型的时候，是不是可以限定下词的长度，只要长度2以上的高频词汇，是不是再去词token化效果会好些。<br><br>

**我之前也尝试：让jieba先切分下，形成列表，然后再用sentencepiece去训练，这样二者就有种互补的效果，一来减少jieba因为词库的原因导致很多高频词组切开的影响，二来可利用sentencepiece的切分效率。但在实际操作中，并没有实现，不知道是对开源的sentencepiece工具没搞清楚，还是它本事就有这个问题。之前也有朋友遇到同样的问题，与他探讨，目前还是没解决。**

5 结语
=
利用sentencpiece代替分词器进行下游任务，是完全可行的，一来它与正规分词器对比效果是相当的，二来它的切分效率很高，可降低模型在token化消耗的时间，这对在工业上应用是合适的。此外，如果在领域性很强的任务时，多在做multi-label任务，sentencpiece带来的效果应该更明显。当然，上述提到的问题，若能解决，会让sentencpiece在中文处理上更有价值。对此感兴趣的朋友，若有啥问题，可与我私下交流~

9 参考
=
1. [Text classification with CNN and Word2vec](https://github.com/cjymz886/text-cnn)
2. [sentencepiece原理与实践](https://zhuanlan.zhihu.com/p/159200073)
3. [SentencePiece](https://github.com/google/sentencepiece)

![image](https://github.com/cjymz886/sentence-similarity/blob/master/images/%E8%87%AA%E7%84%B6%E8%AF%AD%E8%A8%80%E5%A4%84%E7%90%86%E7%AE%97%E6%B3%95%E4%B8%8E%E5%AE%9E%E8%B7%B5.png)<br>


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

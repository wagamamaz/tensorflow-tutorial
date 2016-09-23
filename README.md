# Selected TensorFlow and Deep Learning Tutorial (精选TensorFlow与深度学习教程)

<div align="center">
  <div class="TensorFlow">
    <img src="https://www.tensorflow.org/images/tf_logo_transp.png" style=": left; margin-left: 5px; margin-bottom: 5px;"><br><br>
  </div>
</div>

## TensorFlow 

 - [TensorFlow Official Deep Learning Tutorial](https://www.tensorflow.org/versions/master/tutorials/index.html) [[中文]](http://wiki.jikexueyuan.com/project/tensorflow-zh/).
 - MLP with Dropout [TensorFlow](https://www.tensorflow.org/versions/master/tutorials/mnist/beginners/index.html) [[中文]](http://wiki.jikexueyuan.com/project/tensorflow-zh/tutorials/mnist_beginners.html)  [TensorLayer](http://tensorlayer.readthedocs.io/en/latest/user/tutorial.html#tensorlayer-is-simple) [[中文]](http://tensorlayercn.readthedocs.io/zh/latest/user/tutorial.html#tensorlayer)
 - Autoencoder [TensorLayer](http://tensorlayercn.readthedocs.io/zh/latest/user/tutorial.html#tensorlayer) [[中文]](http://tensorlayercn.readthedocs.io/zh/latest/user/tutorial.html#denoising-autoencoder)
 - Convolutional Neural Network [TensorFlow](https://www.tensorflow.org/versions/master/tutorials/mnist/pros/index.html) [[中文]](http://wiki.jikexueyuan.com/project/tensorflow-zh/tutorials/mnist_pros.html)  [TensorLayer](http://tensorlayer.readthedocs.io/en/latest/user/tutorial.html#convolutional-neural-network-cnn) [[中文]](http://tensorlayercn.readthedocs.io/zh/latest/user/tutorial.html#convolutional-neural-network)
 - Recurrent Neural Network [TensorFlow](https://www.tensorflow.org/versions/master/tutorials/recurrent/index.html#recurrent-neural-networks) [[中文]](http://wiki.jikexueyuan.com/project/tensorflow-zh/tutorials/recurrent.html)  [TensorLayer](http://tensorlayer.readthedocs.io/en/latest/user/tutorial.html#understand-lstm) [[中文]](http://tensorlayercn.readthedocs.io/zh/latest/user/tutorial.html#lstm)
 - Deep Reinforcement Learning [TensorLayer](http://tensorlayer.readthedocs.io/en/latest/user/tutorial.html#understand-reinforcement-learning) [[中文]](http://tensorlayercn.readthedocs.io/zh/latest/user/tutorial.html#id13)
 - Sequence to Sequence [TensorFlow](https://www.tensorflow.org/versions/master/tutorials/seq2seq/index.html#sequence-to-sequence-models)  [TensorLayer](http://tensorlayer.readthedocs.io/en/latest/user/tutorial.html#understand-translation)[[中文]](http://tensorlayercn.readthedocs.io/zh/latest/user/tutorial.html#id30)
 - Word Embedding [TensorFlow](https://www.tensorflow.org/versions/master/tutorials/word2vec/index.html#vector-representations-of-words) [[中文]](http://wiki.jikexueyuan.com/project/tensorflow-zh/tutorials/word2vec.html)  [TensorLayer](http://tensorlayer.readthedocs.io/en/latest/user/tutorial.html#understand-word-embedding) [[中文]](http://tensorlayercn.readthedocs.io/zh/latest/user/tutorial.html#word-embedding)
 
## Deep Learning Reading List

 - [MIT Deep Learning Book](http://www.deeplearningbook.org)
 - [Karpathy Blog](http://karpathy.github.io)
 - [Stanford UFLDL Tutorials](http://deeplearning.stanford.edu/tutorial/)
 - [Colah's Blog - Word Embedding](http://colah.github.io/posts/2014-07-NLP-RNNs-Representations/) [[中文]](http://dataunion.org/9331.html)
 - [Colah's Blog - Understand LSTN](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) [[门函数]](http://mp.weixin.qq.com/s?__biz=MzI3NDExNDY3Nw==&mid=2649764821&idx=1&sn=dd325565b40fcbad6e90a9398414dede&scene=2&srcid=0505U2iFJ7tfXgB8yPfNkwrA&from=timeline&isappinstalled=0#wechat_redirect)
 
## Selected Repositories
 - [jtoy/awesome-tensorflow](https://github.com/jtoy/awesome-tensorflow)
 - [nlintz/TensorFlow-Tutoirals](https://github.com/nlintz/TensorFlow-Tutorials)
 - [adatao/tensorspark](https://github.com/adatao/tensorspark)
 - [ry/tensorflow-resnet](https://github.com/ry/tensorflow-resnet)


## Examples

### Basics


 - Multi-layer perceptron (MNIST). A multi-layer perceptron implementation for MNIST classification task, see ``tutorial_mnist_simple.py`` [here](https://github.com/zsdonghao/tensorlayer).

### Computer Vision

 - Denoising Autoencoder (MNIST). A multi-layer perceptron implementation for MNIST classification task, see ``tutorial_mnist.py`` [here](https://github.com/zsdonghao/tensorlayer).
 - Stacked Denoising Autoencoder and Fine-Tuning (MNIST). A multi-layer perceptron implementation for MNIST classification task, see ``tutorial_mnist.py`` [here](https://github.com/zsdonghao/tensorlayer).
 - Convolutional Network (MNIST). A Convolutional neural network implementation for classifying MNIST dataset, see ``tutorial_mnist.py`` [here](https://github.com/zsdonghao/tensorlayer).
 - Convolutional Network (CIFAR-10). A Convolutional neural network implementation for classifying CIFAR-10 dataset, see ``tutorial_cifar10.py`` [here](https://github.com/zsdonghao/tensorlayer).
 - VGG 16 (ImageNet). A Convolutional neural network implementation for classifying ImageNet dataset, see ``tutorial_vgg16.py`` [here](https://github.com/zsdonghao/tensorlayer).
 - VGG 19 (ImageNet). A Convolutional neural network implementation for classifying ImageNet dataset, see ``tutorial_vgg19.py`` [here](https://github.com/zsdonghao/tensorlayer).


### Natural Language Processing
 - Recurrent Neural Network (LSTM). Apply multiple LSTM to PTB dataset for language modeling, see ``tutorial_ptb_lstm.py``  [here](https://github.com/zsdonghao/tensorlayer).
 - Word Embedding - Word2vec. Train a word embedding matrix, see ``tutorial_word2vec_basic.py`` [here](https://github.com/zsdonghao/tensorlayer).
 - Restore Embedding matrix. Restore a pre-train embedding matrix, see ``tutorial_generate_text.py`` [here](https://github.com/zsdonghao/tensorlayer).
 - Text Generation. Generates new text scripts, using LSTM network, see ``tutorial_generate_text.py`` [here](https://github.com/zsdonghao/tensorlayer).
 - Machine Translation (WMT). Translate English to French. Apply Attention mechanism and Seq2seq to WMT English-to-French translation data, see ``tutorial_translate.py`` [here](https://github.com/zsdonghao/tensorlayer).

### Reinforcement Learning

 - Deep Reinforcement Learning - Pong Game. Teach a machine to play Pong games, see ``tutorial_atari_pong.py`` [here](https://github.com/zsdonghao/tensorlayer).

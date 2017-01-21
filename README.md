# List of my favorite deep learning papers

Rough list of my favorite deep learning research papers, useful for revisiting topics or for reference.


## Recurrent Neural Networks

- [Bidirectional Recurrent Neural Networks](http://www.di.ufpe.br/~fnj/RNA/bibliografia/BRNN.pdf) (Better classifications with RNNs!)
- [Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation](https://arxiv.org/pdf/1406.1078v3.pdf) (Two networks in one combined into a seq2seq (sequence to sequence) Encoder–Decoder architecture. RNN Encoder–Decoder with 1000 hidden units. Adadelta optimizer.)
- [Sequence to Sequence Learning
with Neural Networks](http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf) (4 stacked LSTM cells of 1000 hidden size with reversed input sentences, and with beam search, on the WMT’14 English to French dataset)
- [Exploring the Limits of Language Modeling](https://arxiv.org/pdf/1602.02410.pdf) (Nice recursive models using word level LSTMs on character level CNN using an overkill amount of GPU power)
- [Exploring the Depths of Recurrent Neural Networks with Stochastic Residual Learning](https://cs224d.stanford.edu/reports/PradhanLongpre.pdf) (Basically, residual connections can be better than stacked RNNs in the presented case of sentiment analysis)
- [Neural Turing Machines](https://arxiv.org/pdf/1410.5401v2.pdf) (Outstanding for implementing simple neural algorithms with seemingly good generalisation)
- [Teaching Machines to Read and Comprehend](https://arxiv.org/pdf/1506.03340v3.pdf) (A very interesting and creative work about textual question answering, there is something to do with that)
- [Pixel Recurrent Neural Networks](https://arxiv.org/pdf/1601.06759.pdf) (Nice for photoshop-like "content aware fill" to fill missing patches in images)
- [Adaptive Computation Time for Recurrent Neural Networks](https://arxiv.org/pdf/1603.08983v4.pdf) (Very interesting, I would love to see how well would it combines to Neural Turing Machines. Interesting interactive visualizations on the subject can be found [here](http://distill.pub/2016/augmented-rnns/).)
- [Hybrid computing using a neural network with dynamic external memory](http://www.nature.com/articles/nature20101.epdf?author_access_token=ImTXBI8aWbYxYQ51Plys8NRgN0jAjWel9jnR3ZoTv0MggmpDmwljGswxVdeocYSurJ3hxupzWuRNeGvvXnoO8o4jTJcnAyhGuZzXJ1GEaD-Z7E6X_a9R-xqJ9TfJWBqz) (Improvements on differientable memory based on NTM: now differentiable neural computer)


## Convolutional Neural Networks

- [What is the Best Multi-Stage Architecture for Object Recognition?](http://yann.lecun.com/exdb/publis/pdf/jarrett-iccv-09.pdf) (For the use of "local contrast normalization")
- [ImageNet Classification with Deep Convolutional Neural Networks](http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf) (AlexNet, 2012 ILSVRC, breakthrough of the ReLU activation function)
- [Visualizing and Understanding Convolutional Networks](https://arxiv.org/pdf/1311.2901v3.pdf) (For the "deconvnet layer")
- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/pdf/1409.1556v6.pdf) (For the idea of stacking multiple 3x3 conv+ReLU before pooling for a bigger filter size with few parameters, also there is a nice table for "ConvNet Configuration")
- [Going Deeper with Convolutions](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Szegedy_Going_Deeper_With_2015_CVPR_paper.pdf) (GoogLeNet: Appearance of "Inception" layers/modules, the idea is of parallelizing conv layers into many mini-conv of different size with "same" padding, concatenated on depth)
- [Highway Networks](https://arxiv.org/pdf/1505.00387v2.pdf) (Highway networks: residual connections)
- [Batch Normalization: Accelerating Deep Network Training b
y Reducing Internal Covariate Shift](https://arxiv.org/pdf/1502.03167v3.pdf) (Batch normalization (BN): to normalize a layer's output by also summing over the entire batch, and then performing a linear rescaling and shifting of a certain trainable amount)
- [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385v1.pdf) (Very deep residual layers with batch normalization layers - a.k.a. "how to overfit any vision dataset with too many layers")
- [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/pdf/1602.07261v2.pdf) (For improving GoogLeNet with residual connections)
- [WaveNet: a Generative Model for Raw Audio](https://arxiv.org/pdf/1609.03499v2.pdf) (Epic raw voice/music generation with new architectures based on dilated causal convolutions to capture more audio lenght)

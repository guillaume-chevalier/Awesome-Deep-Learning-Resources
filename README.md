# [Awesome Deep Learning Resources](https://github.com/guillaume-chevalier/Awesome-Deep-Learning-Resources) [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

This is a rough list of my favorite deep learning resources. It has been useful to me for learning how to do deep learning, I use it for revisiting topics or for reference.
I ([Guillaume Chevalier](https://github.com/guillaume-chevalier)) have built this list and got through all of the content listed here, carefully.


## Contents

- [Trends](#trends)
- [Online classes](#online-classes)
- [Books](#books)
- [Posts and Articles](#posts-and-articles)
- [Practical resources](#practical-resources)
  - [Librairies and Implementations](#librairies-and-implementations)
  - [Some Datasets](#some-datasets)
- [Other Math Theory](#other-math-theory)
  - [Gradient Descent Algorithms and optimization](#gradient-descent-algorithms-and-optimization)
  - [Complex Numbers & Digital Signal Processing](#complex-numbers-and-digital-signal-processing)
- [Papers](#papers)
  - [Recurrent Neural Networks](#recurrent-neural-networks)
  - [Convolutional Neural Networks](#convolutional-neural-networks)
  - [Attention Mechanisms](#attention-mechanisms)
  - [Other](#other)
- [YouTube and Videos](#youtube)
- [Misc. Hubs and Links](#misc-hubs-and-links)
- [License](#license)

<a name="trends" />

## Trends

Here are the all-time [Google Trends](https://www.google.ca/trends/explore?date=all&q=machine%20learning,deep%20learning,data%20science,computer%20programming), from 2004 up to now, September 2017:
<p align="center">
  <img src="google_trends.png" width="792" height="424" />
</p>

You might also want to look at Andrej Karpathy's [new post](https://medium.com/@karpathy/a-peek-at-trends-in-machine-learning-ab8a1085a106) about trends in Machine Learning research.

I believe that Deep learning is the key to make computers think more like humans, and has a lot of potential. Some hard automation tasks can be solved easily with that while this was impossible to achieve earlier with classical algorithms.

Moore's Law about exponential progress rates in computer science hardware is now more affecting GPUs than CPUs because of physical limits on how tiny an atomic transistor can be. We are shifting toward parallel architectures
[[read more](https://www.quora.com/Does-Moores-law-apply-to-GPUs-Or-only-CPUs)]. Deep learning exploits parallel architectures as such under the hood by using GPUs. On top of that, deep learning algorithms may use Quantum Computing and apply to machine-brain interfaces in the future.

I find that the key of intelligence and cognition is a very interesting subject to explore and is not yet well understood. Those technologies are promising.


<a name="online-classes" />

## Online Classes

- [Machine Learning by Andrew Ng on Coursera](https://www.coursera.org/learn/machine-learning) - Renown entry-level online class with [certificate](https://www.coursera.org/account/accomplishments/verify/DXPXHYFNGKG3). Taught by: Andrew Ng, Associate Professor, Stanford University; Chief Scientist, Baidu; Chairman and Co-founder, Coursera.
- [Deep Learning Specialization by Andrew Ng on Coursera](https://www.coursera.org/specializations/deep-learning) - New series of 5 Deep Learning courses by Andrew Ng, now with Python rather than Matlab/Octave, and which leads to a [specialization certificate](https://www.coursera.org/account/accomplishments/specialization/U7VNC3ZD9YD8).
- [Deep Learning by Google](https://www.udacity.com/course/deep-learning--ud730) - Good intermediate to advanced-level course covering high-level deep learning concepts, I found it helps to get creative once the basics are acquired.
- [Machine Learning for Trading by Georgia Tech](https://www.udacity.com/course/machine-learning-for-trading--ud501) - Interesting class for acquiring basic knowledge of machine learning applied to trading and some AI and finance concepts. I especially liked the section on Q-Learning.
- [Neural networks class by Hugo Larochelle, Université de Sherbrooke](https://www.youtube.com/playlist?list=PL6Xpj9I5qXYEcOhn7TqghAJ6NAPrNmUBH) - Interesting class about neural networks available online for free by Hugo Larochelle, yet I have watched a few of those videos.
- [GLO-4030/7030 Apprentissage par réseaux de neurones profonds](https://ulaval-damas.github.io/glo4030/) - This is a class given by Philippe Giguère, Professor at University Laval. I especially found awesome its rare visualization of the multi-head attention mechanism, which can be contemplated at the [slide 28 of week 13's class](http://www2.ift.ulaval.ca/~pgiguere/cours/DeepLearning/09-Attention.pdf).
- [Deep Learning & Recurrent Neural Networks (DL&RNN)](https://www.neuraxio.com/en/time-series-solution) - The most richly dense, accelerated course on the topic of Deep Learning & Recurrent Neural Networks (scroll at the end).

<a name="books" />

## Books

- [Clean Code](https://www.amazon.ca/Clean-Code-Handbook-Software-Craftsmanship/dp/0132350882) - Get back to the basics you fool! Learn how to do Clean Code for your career. This is by far the best book I've read even if this list is related to Deep Learning.
- [Clean Coder](https://www.amazon.ca/Clean-Coder-Conduct-Professional-Programmers/dp/0137081073) - Learn how to be professional as a coder and how to interact with your manager. This is important for any coding career.
- [How to Create a Mind](https://www.amazon.com/How-Create-Mind-Thought-Revealed/dp/B009VSFXZ4) - The audio version is nice to listen to while commuting. This book is motivating about reverse-engineering the mind and thinking on how to code AI.
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/index.html) - This book covers many of the core concepts behind neural networks and deep learning.
- [Deep Learning - An MIT Press book](http://www.deeplearningbook.org/) - Yet halfway through the book, it contains satisfying math content on how to think about actual deep learning.
- [Some other books I have read](https://books.google.ca/books?hl=en&as_coll=4&num=100&uid=103409002069648430166&source=gbs_slider_cls_metadata_4_mylibrary_title) - Some books listed here are less related to deep learning but are still somehow relevant to this list.

<a name="posts-and-articles" />

## Posts and Articles

- [Predictions made by Ray Kurzweil](https://en.wikipedia.org/wiki/Predictions_made_by_Ray_Kurzweil) - List of mid to long term futuristic predictions made by Ray Kurzweil.
- [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) - MUST READ post by Andrej Karpathy - this is what motivated me to learn RNNs, it demonstrates what it can achieve in the most basic form of NLP.
- [Neural Networks, Manifolds, and Topology](http://colah.github.io/posts/2014-03-NN-Manifolds-Topology/) - Fresh look on how neurons map information.
- [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) - Explains the LSTM cells' inner workings, plus, it has interesting links in conclusion.
- [Attention and Augmented Recurrent Neural Networks](http://distill.pub/2016/augmented-rnns/) - Interesting for visual animations, it is a nice intro to attention mechanisms as an example.
- [Recommending music on Spotify with deep learning](http://benanne.github.io/2014/08/05/spotify-cnns.html) - Awesome for doing clustering on audio - post by an intern at Spotify.
- [Announcing SyntaxNet: The World’s Most Accurate Parser Goes Open Source](https://research.googleblog.com/2016/05/announcing-syntaxnet-worlds-most.html) - Parsey McParseface's birth, a neural syntax tree parser.
- [Improving Inception and Image Classification in TensorFlow](https://research.googleblog.com/2016/08/improving-inception-and-image.html) - Very interesting CNN architecture (e.g.: the inception-style convolutional layers is promising and efficient in terms of reducing the number of parameters).
- [WaveNet: A Generative Model for Raw Audio](https://deepmind.com/blog/wavenet-generative-model-raw-audio/) - Realistic talking machines: perfect voice generation.
- [François Chollet's Twitter](https://twitter.com/fchollet) - Author of Keras - has interesting Twitter posts and innovative ideas.
- [Neuralink and the Brain’s Magical Future](http://waitbutwhy.com/2017/04/neuralink.html) - Thought provoking article about the future of the brain and brain-computer interfaces.
- [Migrating to Git LFS for Developing Deep Learning Applications with Large Files](http://vooban.com/en/tips-articles-geek-stuff/migrating-to-git-lfs-for-developing-deep-learning-applications-with-large-files/) - Easily manage huge files in your private Git projects.
- [The future of deep learning](https://blog.keras.io/the-future-of-deep-learning.html) - François Chollet's thoughts on the future of deep learning.
- [Discover structure behind data with decision trees](http://vooban.com/en/tips-articles-geek-stuff/discover-structure-behind-data-with-decision-trees/) - Grow decision trees and visualize them, infer the hidden logic behind data.
- [Hyperopt tutorial for Optimizing Neural Networks’ Hyperparameters](http://vooban.com/en/tips-articles-geek-stuff/hyperopt-tutorial-for-optimizing-neural-networks-hyperparameters/) - Learn to slay down hyperparameter spaces automatically rather than by hand.
- [Estimating an Optimal Learning Rate For a Deep Neural Network](https://medium.com/@surmenok/estimating-optimal-learning-rate-for-a-deep-neural-network-ce32f2556ce0) - Clever trick to estimate an optimal learning rate prior any single full training.
 - [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) - Good for understanding the "Attention Is All You Need" (AIAYN) paper. 
 - [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - Also good for understanding the "Attention Is All You Need" (AIAYN) paper.
 - [Improving Language Understanding with Unsupervised Learning](https://blog.openai.com/language-unsupervised/) - SOTA across many NLP tasks from unsupervised pretraining on huge corpus.
 - [NLP's ImageNet moment has arrived](https://thegradient.pub/nlp-imagenet/) - All hail NLP's ImageNet moment. 
 - [The Illustrated BERT, ELMo, and co. (How NLP Cracked Transfer Learning)](https://jalammar.github.io/illustrated-bert/) - Understand the different approaches used for NLP's ImageNet moment. 
 - [Uncle Bob's Principles Of OOD](http://butunclebob.com/ArticleS.UncleBob.PrinciplesOfOod) - Not only the SOLID principles are needed for doing clean code, but the furtherless known REP, CCP, CRP, ADP, SDP and SAP principles are very important for developping huge software that must be bundled in different separated packages.
 - [Why do 87% of data science projects never make it into production?](https://venturebeat.com/2019/07/19/why-do-87-of-data-science-projects-never-make-it-into-production/) - Data is not to be overlooked, and communication between teams and data scientists is important to integrate solutions properly.
 - [The real reason most ML projects fail](https://towardsdatascience.com/what-is-the-main-reason-most-ml-projects-fail-515d409a161f) - Focus on clear business objectives, avoid pivots of algorithms unless you have really clean code, and be able to know when what you coded is "good enough".
 
<a name="practical-resources" />

## Practical Resources

<a name="librairies-and-implementations" />

### Librairies and Implementations
- [Neuraxle, a framwework for machine learning pipelines](https://github.com/Neuraxio/Neuraxle) - The best framework for structuring and deploying your machine learning projects, and which is also compatible with most framework (e.g.: Scikit-Learn, TensorFlow, PyTorch, Keras, and so forth).
- [TensorFlow's GitHub repository](https://github.com/tensorflow/tensorflow) - Most known deep learning framework, both high-level and low-level while staying flexible.
- [skflow](https://github.com/tensorflow/skflow) - TensorFlow wrapper à la scikit-learn.
- [Keras](https://keras.io/) - Keras is another intersting deep learning framework like TensorFlow, it is mostly high-level.
- [carpedm20's repositories](https://github.com/carpedm20) - Many interesting neural network architectures are implemented by the Korean guy Taehoon Kim, A.K.A. carpedm20.
- [carpedm20/NTM-tensorflow](https://github.com/carpedm20/NTM-tensorflow) - Neural Turing Machine TensorFlow implementation.
- [Deep learning for lazybones](http://oduerr.github.io/blog/2016/04/06/Deep-Learning_for_lazybones) - Transfer learning tutorial in TensorFlow for vision from high-level embeddings of a pretrained CNN, AlexNet 2012.
- [LSTM for Human Activity Recognition (HAR)](https://github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition) - Tutorial of mine on using LSTMs on time series for classification.
- [Deep stacked residual bidirectional LSTMs for HAR](https://github.com/guillaume-chevalier/HAR-stacked-residual-bidir-LSTMs) - Improvements on the previous project.
- [Sequence to Sequence (seq2seq) Recurrent Neural Network (RNN) for Time Series Prediction](https://github.com/guillaume-chevalier/seq2seq-signal-prediction) - Tutorial of mine on how to predict temporal sequences of numbers - that may be multichannel.
- [Hyperopt for a Keras CNN on CIFAR-100](https://github.com/guillaume-chevalier/Hyperopt-Keras-CNN-CIFAR-100) - Auto (meta) optimizing a neural net (and its architecture) on the CIFAR-100 dataset.
- [ML / DL repositories I starred](https://github.com/guillaume-chevalier?direction=desc&page=1&q=machine+OR+deep+OR+learning+OR+rnn+OR+lstm+OR+cnn&sort=stars&tab=stars&utf8=%E2%9C%93) - GitHub is full of nice code samples & projects.
- [Smoothly Blend Image Patches](https://github.com/guillaume-chevalier/Smoothly-Blend-Image-Patches) - Smooth patch merger for [semantic segmentation with a U-Net](https://vooban.com/en/tips-articles-geek-stuff/satellite-image-segmentation-workflow-with-u-net/).
- [Self Governing Neural Networks (SGNN): the Projection Layer](https://github.com/guillaume-chevalier/SGNN-Self-Governing-Neural-Networks-Projection-Layer) - With this, you can use words in your deep learning models without training nor loading embeddings.
- [Neuraxle](https://github.com/Neuraxio/Neuraxle) - Neuraxle is a Machine Learning (ML) library for building neat pipelines, providing the right abstractions to both ease research, development, and deployment of your ML applications.
- [Clean Machine Learning, a Coding Kata](https://github.com/Neuraxio/Kata-Clean-Machine-Learning-From-Dirty-Code) - Learn the good design patterns to use for doing Machine Learning the good way, by practicing.

<a name="some-datasets" />

### Some Datasets

Those are resources I have found that seems interesting to develop models onto.

- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets.html) - TONS of datasets for ML.
- [Cornell Movie--Dialogs Corpus](http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html) - This could be used for a chatbot.
- [SQuAD The Stanford Question Answering Dataset](https://rajpurkar.github.io/SQuAD-explorer/) - Question answering dataset that can be explored online, and a list of models performing well on that dataset.
- [LibriSpeech ASR corpus](http://www.openslr.org/12/) - Huge free English speech dataset with balanced genders and speakers, that seems to be of high quality.
- [Awesome Public Datasets](https://github.com/caesar0301/awesome-public-datasets) - An awesome list of public datasets.
- [SentEval: An Evaluation Toolkit for Universal Sentence Representations](https://arxiv.org/abs/1803.05449) - A Python framework to benchmark your sentence representations on many datasets (NLP tasks). 
- [ParlAI: A Dialog Research Software Platform](https://arxiv.org/abs/1705.06476) - Another Python framework to benchmark your sentence representations on many datasets (NLP tasks).


<a name="other-math-theory" />

## Other Math Theory

<a name="gradient-descent-algorithms-and-optimization" />

### Gradient Descent Algorithms & Optimization Theory

- [Neural Networks and Deep Learning, ch.2](http://neuralnetworksanddeeplearning.com/chap2.html) - Overview on how does the backpropagation algorithm works.
- [Neural Networks and Deep Learning, ch.4](http://neuralnetworksanddeeplearning.com/chap4.html) - A visual proof that neural nets can compute any function.
- [Yes you should understand backprop](https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b#.mr5wq61fb) - Exposing backprop's caveats and the importance of knowing that while training models.
- [Artificial Neural Networks: Mathematics of Backpropagation](http://briandolhansky.com/blog/2013/9/27/artificial-neural-networks-backpropagation-part-4) - Picturing backprop, mathematically.
- [Deep Learning Lecture 12: Recurrent Neural Nets and LSTMs](https://www.youtube.com/watch?v=56TYLaQN4N8) - Unfolding of RNN graphs is explained properly, and potential problems about gradient descent algorithms are exposed.
- [Gradient descent algorithms in a saddle point](http://sebastianruder.com/content/images/2016/09/saddle_point_evaluation_optimizers.gif) - Visualize how different optimizers interacts with a saddle points.
- [Gradient descent algorithms in an almost flat landscape](https://devblogs.nvidia.com/wp-content/uploads/2015/12/NKsFHJb.gif) - Visualize how different optimizers interacts with an almost flat landscape.
- [Gradient Descent](https://www.youtube.com/watch?v=F6GSRDoB-Cg) - Okay, I already listed Andrew NG's Coursera class above, but this video especially is quite pertinent as an introduction and defines the gradient descent algorithm.
- [Gradient Descent: Intuition](https://www.youtube.com/watch?v=YovTqTY-PYY) - What follows from the previous video: now add intuition.
- [Gradient Descent in Practice 2: Learning Rate](https://www.youtube.com/watch?v=gX6fZHgfrow) - How to adjust the learning rate of a neural network.
- [The Problem of Overfitting](https://www.youtube.com/watch?v=u73PU6Qwl1I) - A good explanation of overfitting and how to address that problem.
- [Diagnosing Bias vs Variance](https://www.youtube.com/watch?v=ewogYw5oCAI) - Understanding bias and variance in the predictions of a neural net and how to address those problems.
- [Self-Normalizing Neural Networks](https://arxiv.org/pdf/1706.02515.pdf) - Appearance of the incredible SELU activation function.
- [Learning to learn by gradient descent by gradient descent](https://arxiv.org/pdf/1606.04474.pdf) - RNN as an optimizer: introducing the L2L optimizer, a meta-neural network.

<a name="complex-numbers-and-digital-signal-processing" />

### Complex Numbers & Digital Signal Processing

Okay, signal processing might not be directly related to deep learning, but studying it is interesting to have more intuition in developing neural architectures based on signal.

- [Window Functions](https://en.wikipedia.org/wiki/Window_function) - Wikipedia page that lists some of the known window functions - note that the [Hann-Poisson window](https://en.wikipedia.org/wiki/Window_function#Hann%E2%80%93Poisson_window) is specially interesting for greedy hill-climbing algorithms (like gradient descent for example). 
- [MathBox, Tools for Thought Graphical Algebra and Fourier Analysis](https://acko.net/files/gltalks/toolsforthought/) - New look on Fourier analysis.
- [How to Fold a Julia Fractal](http://acko.net/blog/how-to-fold-a-julia-fractal/) - Animations dealing with complex numbers and wave equations.
- [Animate Your Way to Glory, Math and Physics in Motion](http://acko.net/blog/animate-your-way-to-glory/) - Convergence methods in physic engines, and applied to interaction design.
- [Animate Your Way to Glory - Part II, Math and Physics in Motion](http://acko.net/blog/animate-your-way-to-glory-pt2/) - Nice animations for rotation and rotation interpolation with Quaternions, a mathematical object for handling 3D rotations.
- [Filtering signal, plotting the STFT and the Laplace transform](https://github.com/guillaume-chevalier/filtering-stft-and-laplace-transform) - Simple Python demo on signal processing.


<a name="papers" />

## Papers

<a name="recurrent-neural-networks" />

### Recurrent Neural Networks

- [Deep Learning in Neural Networks: An Overview](https://arxiv.org/pdf/1404.7828v4.pdf) - You_Again's summary/overview of deep learning, mostly about RNNs.
- [Bidirectional Recurrent Neural Networks](http://www.di.ufpe.br/~fnj/RNA/bibliografia/BRNN.pdf) - Better classifications with RNNs with bidirectional scanning on the time axis.
- [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/pdf/1406.1078v3.pdf) - Two networks in one combined into a seq2seq (sequence to sequence) Encoder-Decoder architecture. RNN Encoder–Decoder with 1000 hidden units. Adadelta optimizer.
- [Sequence to Sequence Learning with Neural Networks](http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf) - 4 stacked LSTM cells of 1000 hidden size with reversed input sentences, and with beam search, on the WMT’14 English to French dataset.
- [Exploring the Limits of Language Modeling](https://arxiv.org/pdf/1602.02410.pdf) - Nice recursive models using word-level LSTMs on top of a character-level CNN using an overkill amount of GPU power.
- [Neural Machine Translation and Sequence-to-sequence Models: A Tutorial](https://arxiv.org/pdf/1703.01619.pdf) - Interesting overview of the subject of NMT, I mostly read part 8 about RNNs with attention as a refresher.
- [Exploring the Depths of Recurrent Neural Networks with Stochastic Residual Learning](https://cs224d.stanford.edu/reports/PradhanLongpre.pdf) - Basically, residual connections can be better than stacked RNNs in the presented case of sentiment analysis.
- [Pixel Recurrent Neural Networks](https://arxiv.org/pdf/1601.06759.pdf) - Nice for photoshop-like "content aware fill" to fill missing patches in images.
- [Adaptive Computation Time for Recurrent Neural Networks](https://arxiv.org/pdf/1603.08983v4.pdf) - Let RNNs decide how long they compute. I would love to see how well would it combines to Neural Turing Machines. Interesting interactive visualizations on the subject can be found [here](http://distill.pub/2016/augmented-rnns/).

<a name="convolutional-neural-networks" />

### Convolutional Neural Networks

- [What is the Best Multi-Stage Architecture for Object Recognition?](http://yann.lecun.com/exdb/publis/pdf/jarrett-iccv-09.pdf) - Awesome for the use of "local contrast normalization".
- [ImageNet Classification with Deep Convolutional Neural Networks](http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf) - AlexNet, 2012 ILSVRC, breakthrough of the ReLU activation function.
- [Visualizing and Understanding Convolutional Networks](https://arxiv.org/pdf/1311.2901v3.pdf) - For the "deconvnet layer".
- [Fast and Accurate Deep Network Learning by Exponential Linear Units](https://arxiv.org/pdf/1511.07289v1.pdf) - ELU activation function for CIFAR vision tasks.
- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/pdf/1409.1556v6.pdf) - Interesting idea of stacking multiple 3x3 conv+ReLU before pooling for a bigger filter size with just a few parameters. There is also a nice table for "ConvNet Configuration".
- [Going Deeper with Convolutions](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Szegedy_Going_Deeper_With_2015_CVPR_paper.pdf) - GoogLeNet: Appearance of "Inception" layers/modules, the idea is of parallelizing conv layers into many mini-conv of different size with "same" padding, concatenated on depth.
- [Highway Networks](https://arxiv.org/pdf/1505.00387v2.pdf) - Highway networks: residual connections.
- [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/pdf/1502.03167v3.pdf) - Batch normalization (BN): to normalize a layer's output by also summing over the entire batch, and then performing a linear rescaling and shifting of a certain trainable amount.
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597.pdf) - The U-Net is an encoder-decoder CNN that also has skip-connections, good for image segmentation at a per-pixel level.
- [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385v1.pdf) - Very deep residual layers with batch normalization layers - a.k.a. "how to overfit any vision dataset with too many layers and make any vision model work properly at recognition given enough data".
- [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/pdf/1602.07261v2.pdf) - For improving GoogLeNet with residual connections.
- [WaveNet: a Generative Model for Raw Audio](https://arxiv.org/pdf/1609.03499v2.pdf) - Epic raw voice/music generation with new architectures based on dilated causal convolutions to capture more audio length.
- [Learning a Probabilistic Latent Space of Object Shapes via 3D Generative-Adversarial Modeling](https://arxiv.org/pdf/1610.07584v2.pdf) - 3D-GANs for 3D model generation and fun 3D furniture arithmetics from embeddings (think like word2vec word arithmetics with 3D furniture representations).
- [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://research.fb.com/publications/ImageNet1kIn1h/) - Incredibly fast distributed training of a CNN.
- [Densely Connected Convolutional Networks](https://arxiv.org/pdf/1608.06993.pdf) - Best Paper Award at CVPR 2017, yielding improvements on state-of-the-art performances on CIFAR-10, CIFAR-100 and SVHN datasets, this new neural network architecture is named DenseNet.
- [The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation](https://arxiv.org/pdf/1611.09326.pdf) - Merges the ideas of the U-Net and the DenseNet, this new neural network is especially good for huge datasets in image segmentation.
- [Prototypical Networks for Few-shot Learning](https://arxiv.org/pdf/1703.05175.pdf) - Use a distance metric in the loss to determine to which class does an object belongs to from a few examples.

<a name="attention-mechanisms" />

### Attention Mechanisms

- [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/pdf/1409.0473.pdf) - Attention mechanism for LSTMs! Mostly, figures and formulas and their explanations revealed to be useful to me. I gave a talk on that paper [here](https://www.youtube.com/watch?v=QuvRWevJMZ4).
- [Neural Turing Machines](https://arxiv.org/pdf/1410.5401v2.pdf) - Outstanding for letting a neural network learn an algorithm with seemingly good generalization over long time dependencies. Sequences recall problem.
- [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/pdf/1502.03044.pdf) - LSTMs' attention mechanisms on CNNs feature maps does wonders.
- [Teaching Machines to Read and Comprehend](https://arxiv.org/pdf/1506.03340v3.pdf) - A very interesting and creative work about textual question answering, what a breakthrough, there is something to do with that.
- [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/pdf/1508.04025.pdf) - Exploring different approaches to attention mechanisms.
- [Matching Networks for One Shot Learning](https://arxiv.org/pdf/1606.04080.pdf) - Interesting way of doing one-shot learning with low-data by using an attention mechanism and a query to compare an image to other images for classification.
- [Google’s Neural Machine Translation System: Bridging the Gap between Human and Machine Translation](https://arxiv.org/pdf/1609.08144.pdf) - In 2016: stacked residual LSTMs with attention mechanisms on encoder/decoder are the best for NMT (Neural Machine Translation).
- [Hybrid computing using a neural network with dynamic external memory](http://www.nature.com/articles/nature20101.epdf?author_access_token=ImTXBI8aWbYxYQ51Plys8NRgN0jAjWel9jnR3ZoTv0MggmpDmwljGswxVdeocYSurJ3hxupzWuRNeGvvXnoO8o4jTJcnAyhGuZzXJ1GEaD-Z7E6X_a9R-xqJ9TfJWBqz) - Improvements on differentiable memory based on NTMs: now it is the Differentiable Neural Computer (DNC).
- [Massive Exploration of Neural Machine Translation Architectures](https://arxiv.org/pdf/1703.03906.pdf) - That yields intuition about the boundaries of what works for doing NMT within a framed seq2seq problem formulation.
- [Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram
Predictions](https://arxiv.org/pdf/1712.05884.pdf) - A [WaveNet](https://arxiv.org/pdf/1609.03499v2.pdf) used as a vocoder can be conditioned on generated Mel Spectrograms from the Tacotron 2 LSTM neural network with attention to generate neat audio from text.
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (AIAYN) - Introducing multi-head self-attention neural networks with positional encoding to do sentence-level NLP without any RNN nor CNN - this paper is a must-read (also see [this explanation](http://nlp.seas.harvard.edu/2018/04/03/attention.html) and [this visualization](http://jalammar.github.io/illustrated-transformer/) of the paper). 

<a name="other" />

### Other

- [ProjectionNet: Learning Efficient On-Device Deep Networks Using Neural Projections](https://arxiv.org/abs/1708.00630) - Replace word embeddings by word projections in your deep neural networks, which doesn't require a pre-extracted dictionnary nor storing embedding matrices. 
- [Self-Governing Neural Networks for On-Device Short Text Classification](http://aclweb.org/anthology/D18-1105) - This paper is the sequel to the ProjectionNet just above. The SGNN is elaborated on the ProjectionNet, and the optimizations are detailed more in-depth (also see my [attempt to reproduce the paper in code](https://github.com/guillaume-chevalier/SGNN-Self-Governing-Neural-Networks-Projection-Layer) and watch [the talks' recording](https://vimeo.com/305197775)).
- [Matching Networks for One Shot Learning](https://arxiv.org/abs/1606.04080) - Classify a new example from a list of other examples (without definitive categories) and with low-data per classification task, but lots of data for lots of similar classification tasks - it seems better than siamese networks. To sum up: with Matching Networks, you can optimize directly for a cosine similarity between examples (like a self-attention product would match) which is passed to the softmax directly. I guess that Matching Networks could probably be used as with negative-sampling softmax training in word2vec's CBOW or Skip-gram without having to do any context embedding lookups. 


<a name="youtube" />

## YouTube and Videos

- [Attention Mechanisms in Recurrent Neural Networks (RNNs) - IGGG](https://www.youtube.com/watch?v=QuvRWevJMZ4) - A talk for a reading group on attention mechanisms (Paper: Neural Machine Translation by Jointly Learning to Align and Translate).
- [Tensor Calculus and the Calculus of Moving Surfaces](https://www.youtube.com/playlist?list=PLlXfTHzgMRULkodlIEqfgTS-H1AY_bNtq) - Generalize properly how Tensors work, yet just watching a few videos already helps a lot to grasp the concepts.
- [Deep Learning & Machine Learning (Advanced topics)](https://www.youtube.com/playlist?list=PLlp-GWNOd6m4C_-9HxuHg2_ZeI2Yzwwqt) - A list of videos about deep learning that I found interesting or useful, this is a mix of a bit of everything.
- [Signal Processing Playlist](https://www.youtube.com/playlist?list=PLlp-GWNOd6m6gSz0wIcpvl4ixSlS-HEmr) - A YouTube playlist I composed about DFT/FFT, STFT and the Laplace transform - I was mad about my software engineering bachelor not including signal processing classes (except a bit in the quantum physics class).
- [Computer Science](https://www.youtube.com/playlist?list=PLlp-GWNOd6m7vLOsW20xAJ81-65C-Ys6k) - Yet another YouTube playlist I composed, this time about various CS topics.
- [Siraj's Channel](https://www.youtube.com/channel/UCWN3xxRkmTPmbKwht9FuE5A/videos?view=0&sort=p&flow=grid) - Siraj has entertaining, fast-paced video tutorials about deep learning.
- [Two Minute Papers' Channel](https://www.youtube.com/user/keeroyz/videos?sort=p&view=0&flow=grid) - Interesting and shallow overview of some research papers, for example about WaveNet or Neural Style Transfer.
- [Geoffrey Hinton interview](https://www.coursera.org/learn/neural-networks-deep-learning/lecture/dcm5r/geoffrey-hinton-interview) - Andrew Ng interviews Geoffrey Hinton, who talks about his research and breaktroughs, and gives advice for students.
- [Growing Neat Software Architecture from Jupyter Notebooks](https://www.youtube.com/watch?v=K4QN27IKr0g) - A primer on how to structure your Machine Learning projects when using Jupyter Notebooks.

<a name="misc-hubs-and-links" />

## Misc. Hubs & Links

- [Hacker News](https://news.ycombinator.com/news) - Maybe how I discovered ML - Interesting trends appear on that site way before they get to be a big deal.
- [DataTau](http://www.datatau.com/) - This is a hub similar to Hacker News, but specific to data science.
- [Naver](http://www.naver.com/) - This is a Korean search engine - best used with Google Translate, ironically. Surprisingly, sometimes deep learning search results and comprehensible advanced math content shows up more easily there than on Google search.
- [Arxiv Sanity Preserver](http://www.arxiv-sanity.com/) - arXiv browser with TF/IDF features.


<a name="license" />

## License

[![CC0](http://mirrors.creativecommons.org/presskit/buttons/88x31/svg/cc-zero.svg)](https://creativecommons.org/publicdomain/zero/1.0/)

To the extent possible under law, [Guillaume Chevalier](https://github.com/guillaume-chevalier) has waived all copyright and related or neighboring rights to this work.

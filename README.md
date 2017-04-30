# List of awesome deep learning resources ![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)

Rough list of my favorite deep learning resources, useful for revisiting topics or for reference. I (Guillaume Chevalier) have got through all of the content listed there, carefully, and built this list.

## Trends

Here are the all-time [Google Trends](https://www.google.ca/trends/explore?date=all&q=machine%20learning,deep%20learning,data%20science,computer%20programming), from 2004 up to now, April 2017:
<p align="center">
  <img src="google_trends.png" />
</p>

You might also want to look at Andrej Karpathy's [new post](https://medium.com/@karpathy/a-peek-at-trends-in-machine-learning-ab8a1085a106) about trends in Machine Learning research.

I believe that Deep learning is the key that can make computers think more like humans and has a lot of potential. Some hard automation tasks can be solved easily with that while this was impossible to achieve earlier with classical algorithms.

Moore's Law about exponential progress rates in computer science hardware is now more affecting GPUs than CPUs because of physical limits on how tiny an atomic transistor can be. We are shifting toward parallel architectures
[[read more](https://www.quora.com/Does-Moores-law-apply-to-GPUs-Or-only-CPUs)]. Deep learning exploits parallel architectures as such under the hood by using GPUs. On top of that, deep learning algorithms may be applied to Quantum Computing.

The key of intelligence and cognition is a very interesting subject to explore and is not yet well understood. 


## Online classes

- [Machine Learning by Andrew Ng on Coursera](https://www.coursera.org/learn/machine-learning) (Good entry-level online class with [certificate](https://www.coursera.org/account/accomplishments/verify/DXPXHYFNGKG3). Taught by: Andrew Ng, Associate Professor, Stanford University; Chief Scientist, Baidu; Chairman and Co-founder, Coursera)
- [Deep Learning by Google](https://www.udacity.com/course/deep-learning--ud730) (Very good class covering high-level deep learning concepts, it is the logical next step after Andrew Ng's Machine Learning class)
- [Machine Learning for Trading by Georgia Tech](https://www.udacity.com/course/machine-learning-for-trading--ud501) (Interesting class for acquiring basic knowledge of machine learning applied to trading and some AI and finance concepts)
- [Neural networks class - Université de Sherbrooke](https://www.youtube.com/playlist?list=PL6Xpj9I5qXYEcOhn7TqghAJ6NAPrNmUBH) (Very interesting class available online for free by Hugo Larochelle, yet I have watched a few of those videos)


## Books

- [How to Create a Mind](https://www.amazon.com/How-Create-Mind-Thought-Revealed/dp/B009VSFXZ4) (The audio version is nice to listen to while commuting)
- [Deep Learning - An MIT Press book](http://www.deeplearningbook.org/) (Outstanding, yet halfway through the book)
- [Some other books I have read](https://books.google.ca/books?hl=en&as_coll=4&num=10&uid=103409002069648430166&source=gbs_slider_cls_metadata_4_mylibrary_title) (Some may be less related to deep learning but still somehow relevant)

## Posts and Articles

- [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) (MUST READ post by Andrej Karpathy - this is what motivated me to learn RNNs)
- [Neural Networks, Manifolds, and Topology](http://colah.github.io/posts/2014-03-NN-Manifolds-Topology/) (Fresh look on how neurons map information)
- [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) (Walktrough the LSTM cells' inner workings, plus interesting links in conclusion)
- [Attention and Augmented Recurrent Neural Networks](http://distill.pub/2016/augmented-rnns/) (Interesting for visual animations)
- [Recommending music on Spotify with deep learning](http://benanne.github.io/2014/08/05/spotify-cnns.html) (Awesome for doing clustering on audio - post by an intern at Spotify)
- [Announcing SyntaxNet: The World’s Most Accurate Parser Goes Open Source](https://research.googleblog.com/2016/05/announcing-syntaxnet-worlds-most.html) (Parsey McParseface's birth)
- [Improving Inception and Image Classification in TensorFlow](https://research.googleblog.com/2016/08/improving-inception-and-image.html) (Very interesting CNN architecture)
- [WaveNet: A Generative Model for Raw Audio](https://deepmind.com/blog/wavenet-generative-model-raw-audio/) (Realistic talking machines)
- [François Chollet's Twitter](https://twitter.com/fchollet) (Author of Keras - has interesting Twitter posts)
- [Neuralink and the Brain’s Magical Future](http://waitbutwhy.com/2017/04/neuralink.html) (Very worth reading.)

## Practical resources

### Librairies and Implementations
- [TensorFlow's GitHub repository](https://github.com/tensorflow/tensorflow) (Most known deep learning framework, both flexible and high-level)
- [skflow](https://github.com/tensorflow/skflow) (TensorFlow wrapper à la scikit-learn)
- [Keras](https://keras.io/) (Keras is another very intersting deep learning framework like TensorFlow)
- [carpedm20's repositories](https://github.com/carpedm20) (Many interesting neural network architectures are implemented by the Korean guy Taehoon Kim - A.K.A. carpedm20)
- [carpedm20/NTM-tensorflow](https://github.com/carpedm20/NTM-tensorflow) (Neural Turing Machine TensorFlow implementation)
- [Deep learning for lazybones](http://oduerr.github.io/blog/2016/04/06/Deep-Learning_for_lazybones) (Transfer learning tutorial in TensorFlow for vision from high-level embeddings of a pretrained CNN, AlexNet 2012)
- [LSTM for Human Activity Recognition (HAR)](https://github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition) (Tutorial of mine on using LSTMs on time series for classification)
- [Deep residual bidirectional LSTM for HAR](https://github.com/guillaume-chevalier/HAR-stacked-residual-bidir-LSTMs) (Improvements on the previous project)
- [ML / DL repositories I starred](https://github.com/guillaume-chevalier?direction=desc&page=1&q=machine+OR+deep+OR+learning+OR+rnn+OR+lstm+OR+cnn&sort=stars&tab=stars&utf8=%E2%9C%93) (GitHub is full of nice code samples & projects)

### Some Datasets

Those are resources I have found that seems interesting to develop models onto.

- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets.html) (TONS of datasets for ML)
- [Cornell Movie--Dialogs Corpus](http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html) (Could be used for a chatbot)
- [SQuAD
The Stanford Question Answering Dataset](https://rajpurkar.github.io/SQuAD-explorer/) (Interesting QA dataset)
- [Aligned Hansards of the 36th Parliament of Canada](http://www.isi.edu/natural-language/download/hansard/) (Aligned text chunks useful for FR-EN machine translation)
- [LibriSpeech ASR corpus](http://www.openslr.org/12/) (Huge free English speech dataset with balanced genders and speakers that seems to be of high quality)


## Other Math Theory

### Gradient Descent Algorithms and optimization

- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/chap2.html) (Overview)
- [Yes you should understand backprop](https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b#.mr5wq61fb) (Exposing backprop's caveats)
- [Artificial Neural Networks: Mathematics of Backpropagation](http://briandolhansky.com/blog/2013/9/27/artificial-neural-networks-backpropagation-part-4) (Picturing backprop mathematically)
- [Deep Learning Lecture 12: Recurrent Neural Nets and LSTMs](https://www.youtube.com/watch?v=56TYLaQN4N8) (Unfolding of the RNN graphs is explained properly and expose potential problems about gradient descent algorithms)
- [Gradient descent algorithms in a saddle point](http://sebastianruder.com/content/images/2016/09/saddle_point_evaluation_optimizers.gif) (Interesting image)
- [Gradient descent algorithms in an almost flat landscape](https://devblogs.nvidia.com/wp-content/uploads/2015/12/NKsFHJb.gif) (Complementary interesting image to the previous one)
- [Gradient Descent](https://www.youtube.com/watch?v=eikJboPQDT0) (Okay, I already listed Andrew NG's Coursera class above, but this video is quite pertinent)
- [Gradient Descent: Intuition](https://www.youtube.com/watch?v=Fn8qXpIcdnI) (continued)
- [Gradient Descent in Practice 2: Learning Rate](https://www.youtube.com/watch?v=4WGK1GrFYd8&list=PLLH73N9cB21V_O2JqILVX557BST2cqJw4&index=22) (continued - adjusting learning rate)
- [The Problem of Overfitting](https://www.youtube.com/watch?v=Tl5JB5EuU5o&list=PLLH73N9cB21V_O2JqILVX557BST2cqJw4&index=40) (continued - intro to overfitting)
- [Diagnosing Bias vs Variance](https://www.youtube.com/watch?v=ewogYw5oCAI) (continued - heuristics for optimizing)

### Complex Numbers & Digital Signal Processing

Okay, signal processing might not be directly related to deep learning, but studying it is interesting for have better intuition for developing neural architectures based on signal.

- [Window Functions](https://en.wikipedia.org/wiki/Window_function) (Interesting Wikipedia page listing known window functions)
- [MathBox, Tools for Thought Graphical Algebra and Fourier Analysis](https://acko.net/files/gltalks/toolsforthought/) (New look on Fourier analysis)
- [How to Fold a Julia Fractal](http://acko.net/blog/how-to-fold-a-julia-fractal/) (Very interesting animations dealing with complex numbers and wave equations)
- [Animate Your Way to Glory, Math and Physics in Motion](http://acko.net/blog/animate-your-way-to-glory/) (Look on convergence methods in physic engines and interaction design)
- [Animate Your Way to Glory - Part II, Math and Physics in Motion](http://acko.net/blog/animate-your-way-to-glory-pt2/) (Nice animations for rotation and rotation interpolation with Quaternions)
- [Filtering signal, plotting the STFT and the Laplace transform](https://github.com/guillaume-chevalier/filtering-stft-and-laplace-transform) (Simple Python demo on signal processing)


## Papers

### Recurrent Neural Networks

- [Deep Learning in Neural Networks: An Overview](https://arxiv.org/pdf/1404.7828v4.pdf) (You_Again's DL summary, mostly about RNNs)
- [Bidirectional Recurrent Neural Networks](http://www.di.ufpe.br/~fnj/RNA/bibliografia/BRNN.pdf) (Better classifications with RNNs!)
- [Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation](https://arxiv.org/pdf/1406.1078v3.pdf) (Two networks in one combined into a seq2seq (sequence to sequence) Encoder–Decoder architecture. RNN Encoder–Decoder with 1000 hidden units. Adadelta optimizer.)
- [Sequence to Sequence Learning with Neural Networks](http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf) (4 stacked LSTM cells of 1000 hidden size with reversed input sentences, and with beam search, on the WMT’14 English to French dataset)
- [Exploring the Limits of Language Modeling](https://arxiv.org/pdf/1602.02410.pdf) (Nice recursive models using word level LSTMs on character level CNN using an overkill amount of GPU power)
- [Exploring the Depths of Recurrent Neural Networks with Stochastic Residual Learning](https://cs224d.stanford.edu/reports/PradhanLongpre.pdf) (Basically, residual connections can be better than stacked RNNs in the presented case of sentiment analysis)
- [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/pdf/1409.0473.pdf) (Attention mechanism for LSTMs! Mostly, figures and formulas and their explanations revealed to be useful to me.)
- [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/pdf/1502.03044.pdf) (LSTM attention mechanism on CNNs)
- [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/pdf/1508.04025.pdf) (Exploring different attention mechanisms)
- [Neural Turing Machines](https://arxiv.org/pdf/1410.5401v2.pdf) (Outstanding for implementing simple neural algorithms with seemingly good generalisation)
- [Teaching Machines to Read and Comprehend](https://arxiv.org/pdf/1506.03340v3.pdf) (A very interesting and creative work about textual question answering, there is something to do with that)
- [Pixel Recurrent Neural Networks](https://arxiv.org/pdf/1601.06759.pdf) (Nice for photoshop-like "content aware fill" to fill missing patches in images)
- [Adaptive Computation Time for Recurrent Neural Networks](https://arxiv.org/pdf/1603.08983v4.pdf) (Very interesting, I would love to see how well would it combines to Neural Turing Machines. Interesting interactive visualizations on the subject can be found [here](http://distill.pub/2016/augmented-rnns/).)
- [Hybrid computing using a neural network with dynamic external memory](http://www.nature.com/articles/nature20101.epdf?author_access_token=ImTXBI8aWbYxYQ51Plys8NRgN0jAjWel9jnR3ZoTv0MggmpDmwljGswxVdeocYSurJ3hxupzWuRNeGvvXnoO8o4jTJcnAyhGuZzXJ1GEaD-Z7E6X_a9R-xqJ9TfJWBqz) (Improvements on differientable memory based on NTM: now differentiable neural computer)
- [Google’s Neural Machine Translation System: Bridging the Gap between Human and Machine Translation](https://arxiv.org/pdf/1609.08144.pdf) (In 2016: stacked residual LSTMs with attention mechanisms encoder/decoder are the best for NMT - Neural Machine Translation)
- [Neural Machine Translation and Sequence-to-sequence Models: A Tutorial](https://arxiv.org/pdf/1703.01619.pdf) (Interesting tutorial, I mostly read part 8 about RNNs with attention as a refresher for NMT.)

### Convolutional Neural Networks

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
- [Learning a Probabilistic Latent Space of Object Shapes via 3D Generative-Adversarial Modeling](https://arxiv.org/pdf/1610.07584v2.pdf) (3D-GAN for 3D model generation and fun 3D furniture arithmetics from embeddings)


## YouTube

- [Deep Learning & Machine Learning (Advanced topics)](https://www.youtube.com/playlist?list=PLlp-GWNOd6m4C_-9HxuHg2_ZeI2Yzwwqt) (Awesome videos)
- [Tensor Calculus and the Calculus of Moving Surfaces](https://www.youtube.com/playlist?list=PLlXfTHzgMRULkodlIEqfgTS-H1AY_bNtq) (Generalize properly how Tensors work, yet watching a few videos helps)
- [Signal Processing YouTube Playlist](https://www.youtube.com/playlist?list=PLlp-GWNOd6m6gSz0wIcpvl4ixSlS-HEmr) (A YouTube playlist I composed about DFT/FFT, STFT and the Laplace transform - I was mad about my software engineering bachelor not including signal processing classes)
- [Computer Science](https://www.youtube.com/playlist?list=PLlp-GWNOd6m7vLOsW20xAJ81-65C-Ys6k) (Yet another YouTube playlist I composed, this time about various CS topics just slightly related)
- [Siraj's YouTube Channel](https://www.youtube.com/channel/UCWN3xxRkmTPmbKwht9FuE5A/videos?view=0&sort=p&flow=grid) (Siraj has some entertaining videos)
- [Two Minute Papers' Youtube Channel](https://www.youtube.com/user/keeroyz/videos?sort=p&view=0&flow=grid) (Interesting but very shallow overview of some papers, for example about WaveNet or Neural Style Transfer)

## Misc. Hubs and Links

- [Quora.com](https://www.quora.com/) (Finest question/answer site)
- [Hacker News](https://news.ycombinator.com/news) (Maybe how I discovered ML - Interesting trends appear on that site way before they get to be a big deal)
- [DataTau](http://www.datatau.com/) (A hub similar to Hacker News, but specific to data science)
- [Naver](http://www.naver.com/) (Korean search engine - best used with Google Translate, ironically. Surprisingly, sometimes deep learning search results and comprehensible advanced math content shows up more easily there than on Google search)
- [Arxiv Sanity Preserver](http://www.arxiv-sanity.com/) (arXiv browser with TF/IDF features)

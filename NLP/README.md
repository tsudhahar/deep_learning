# Text Classification: All Tips and Tricks from 5 Kaggle Competitions



In this article, I will discuss some great tips and tricks to improve the performance of your text classification model. These tricks are obtained from solutions of some of Kaggle&#39;s top NLP competitions.

Namely, I&#39;ve gone through:

- [Jigsaw Unintended Bias in Toxicity Classification](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification) – $65,000
- [Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/) – $35,000
- [Quora Insincere Questions Classification](http://kaggle.com/c/quora-insincere-questions-classification) – $25,000
- [Google QUEST Q&amp;A Labeling](https://www.kaggle.com/c/google-quest-challenge) – $25,000
- [TensorFlow 2.0 Question Answering](https://www.kaggle.com/c/tensorflow2-question-answering) – $50,000

and found a ton of great ideas.

Without much lag, let&#39;s begin.

Dealing with larger datasets

One issue you might face in any machine learning competition is the size of your data set. If the size of your data is large, that is 3GB + for Kaggle kernels and more basic laptops you could find it difficult to load and process with limited resources. Here is the link to some of the articles and kernels that I have found useful in such situations.

- Optimize the memory by [reducing the size of some attributes](https://www.kaggle.com/shrutimechlearn/large-data-loading-trick-with-ms-malware-data)
- Use open-source libraries such as[ Dask to read and manipulate the data](https://www.kaggle.com/yuliagm/how-to-work-with-big-datasets-on-16g-ram-dask), it performs parallel computing and saves up memory space
- Use [cudf](https://github.com/rapidsai/cudf)
- Convert data to [parquet](https://arrow.apache.org/docs/python/parquet.html) format
- Convert data to [feather](https://medium.com/@snehotosh.banerjee/feather-a-fast-on-disk-format-for-r-and-python-data-frames-de33d0516b03) format

Small datasets and external data

But, what can one do if the dataset is small? Let&#39;s see some techniques to tackle this situation.

One way to increase the performance of any machine learning model is to use some external data frame that contains some variables that influence the predicate variable.

Let&#39;s see some of the external datasets.

- Use of [squad](https://rajpurkar.github.io/SQuAD-explorer/) data for Question Answering tasks
- Other [datasets](http://nlpprogress.com/english/question_answering.html) for QA tasks
- Wikitext long term dependency language modeling [dataset](https://www.salesforce.com/products/einstein/ai-research/the-wikitext-dependency-language-modeling-dataset/)
- [Stackexchange data](https://archive.org/download/stackexchange)
- Prepare a dictionary of commonly misspelled words and corrected words.
- Use of [helper datasets](https://www.kaggle.com/kyakovlev/jigsaw-general-helper-public) for cleaning
- [Pseudo labeling](https://www.kaggle.com/cdeotte/pseudo-labeling-qda-0-969/) is the process of adding confidently predicted test data to your training data
- Use different data [sampling methods](https://www.kaggle.com/shahules/tackling-class-imbalance)
- Text augmentation by [Exchanging words with synonym](https://arxiv.org/pdf/1502.01710.pdf)[s](https://arxiv.org/pdf/1502.01710.pdf)
- Text augmentation by [noising in RNN](https://arxiv.org/pdf/1703.02573.pdf)
- Text augmentation by [translation to other languages and back](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/48038)

Data exploration and gaining insights

Data exploration always helps to better understand the data and gain insights from it. Before starting to develop machine learning models, top competitors always read/do a lot of exploratory data analysis for the data. This helps in feature engineering and cleaning of the data.

- Twitter data [exploration methods](https://www.kaggle.com/jagangupta/stop-the-s-toxic-comments-eda)
- Simple [EDA for tweets](https://www.kaggle.com/nz0722/simple-eda-text-preprocessing-jigsaw)
- [EDA](https://www.kaggle.com/tunguz/just-some-simple-eda) for Quora data
- [EDA](https://www.kaggle.com/kailex/r-eda-for-q-gru) in  R for Quora data
- Complete [EDA](https://www.kaggle.com/codename007/start-from-here-quest-complete-eda-fe) with stack exchange data
- My previous article on [EDA for natural language processing](https://neptune.ai/blog/exploratory-data-analysis-natural-language-processing-tools)

Data cleaning

Data cleaning is one of the important and integral parts of any NLP problem. Text data always needs some preprocessing and cleaning before we can represent it in a suitable form.

- Use this [notebook](https://www.kaggle.com/sudalairajkumar/getting-started-with-text-preprocessing) to clean social media data
- [Data cleaning](https://www.kaggle.com/kyakovlev/preprocessing-bert-public) for BERT
- Use [textblob](https://textblob.readthedocs.io/en/dev/quickstart.html) to correct misspellings
- [Cleaning](https://www.kaggle.com/theoviel/improve-your-score-with-some-text-preprocessing) for pre-trained embeddings
- [Language detection and translation](https://www.pythonprogramming.in/language-detection-and-translation-using-textblob.html) for multilingual tasks
- Preprocessing for Glove [part 1](https://www.kaggle.com/christofhenkel/how-to-preprocessing-for-glove-part1-eda) and [part 2](https://www.kaggle.com/christofhenkel/how-to-preprocessing-for-glove-part2-usage)
- [Increasing word coverage](https://www.kaggle.com/sunnymarkliu/more-text-cleaning-to-increase-word-coverage) to get more from pre-trained word embeddings

Text representations

Before we feed our text data to the Neural network or ML model, the text input needs to be represented in a suitable format. These representations determine the performance of the model to a large extent.

- Pretrained [Glove](https://nlp.stanford.edu/projects/glove/) vectors
- Pretrained [fasttext](https://fasttext.cc/docs/en/english-vectors.html) vectors
- Pretrained [word2vec](https://radimrehurek.com/gensim/models/word2vec.html) vectors
- My previous article on these[ 3 embeddings](https://neptune.ai/blog/document-classification-small-datasets)
- Combining [pre-trained vectors](https://www.kaggle.com/c/quora-insincere-questions-classification/discussion/71778/). This can help in better representation of text and decreasing OOV words
- [Paragram](https://cogcomp.seas.upenn.edu/page/resource_view/106) embeddings
- [Universal Sentence Encoder](https://tfhub.dev/google/universal-sentence-encoder/1)
- Use USE to generate [sentence-level features](https://www.kaggle.com/abhishek/distilbert-use-features-oof)
- 3 methods to [combine embeddings](https://www.kaggle.com/c/quora-insincere-questions-classification/discussion/71778)

Contextual embeddings models

- [BERT](https://github.com/google-research/bert) Bidirectional Encoder Representations from Transformers
- [GPT](https://github.com/openai/finetune-transformer-lm)
- [Roberta](https://github.com/pytorch/fairseq/tree/master/examples/roberta) a Robustly Optimized BERT
- [Albert](https://github.com/google-research/ALBERT) a Lite BERT for Self-supervised Learning of Language Representations
- [Distilbert](https://github.com/huggingface/transformers/tree/master/examples/distillation) a lighter version of BERT
- [XLNET](https://github.com/zihangdai/xlnet/)

Modeling

Model architecture

Choosing the right architecture is important to develop a proper machine learning model, sequence to sequence models like LSTMs, GRUs perform well in NLP problems and is always worth trying. Stacking 2 layers of LSTM/GRU networks is a common approach.

- [Stacking Bidirectional CuDNNLSTM](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/52644/)
- [Stacking LSTM networks](https://www.kaggle.com/sakami/google-quest-single-lstm/)
- [LSTM and 5 fold Attention](https://www.kaggle.com/christofhenkel/keras-baseline-lstm-attention-5-fold/)
- [Bidirectional LSTM with 1D convolutions](https://www.kaggle.com/c/quora-insincere-questions-classification/discussion/80568/)
- [Unfreeze and tune embeddings](https://www.kaggle.com/c/quora-insincere-questions-classification/discussion/80542/)
- [BiLSTM with Global maxpooling](https://www.kaggle.com/wowfattie/3rd-place)
- [Attention weighted average](https://www.kaggle.com/c/quora-insincere-questions-classification/discussion/80495)
- [GRU+ Capsule network](https://www.kaggle.com/gmhost/gru-capsule)
- [InceptionCNN with flip](https://www.kaggle.com/christofhenkel/inceptioncnn-with-flip)
- [Plain vanilla network with BERT](https://www.kaggle.com/yuval6967/toxic-bert-plain-vanila)
- [CuDNNGRU network](https://www.kaggle.com/taindow/simple-cudnngru-python-keras)
- [TextCNN with pooling layers](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/52719)
- [BERT embeddings with LSTM](https://www.kaggle.com/christofhenkel/bert-embeddings-lstm)
- [Multi-sample dropouts](https://arxiv.org/abs/1905.09788)
- [Siamese transformer network](https://www.kaggle.com/c/google-quest-challenge/discussion/129978)
- [Global Average pooling of hidden layers BERT](https://www.kaggle.com/akensert/bert-base-tf2-0-now-huggingface-transformer)
- [Different Bert based models](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/discussion/92867)
- Distilling BERT — [BERT performance using Logistic Regression](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/discussion/97135)
- [Different learning rates among the layers of BERT](https://medium.com/nvidia-ai/a-guide-to-optimizer-implementation-for-bert-at-scale-8338cc7f45fd)
- [Finetuning Bert for text classification](https://arxiv.org/abs/1905.05583)

Loss functions

Choosing a proper loss function for your NN model really enhances the performance of your model by allowing it to optimize well on the surface.

You can try different loss functions or even write a custom loss function that matches your problem. Some of the popular loss functions are

- [Binary cross-entropy](https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html) for binary classification
- [Categorical cross-entropy](https://peltarion.com/knowledge-center/documentation/modeling-view/build-an-ai-model/loss-functions/categorical-crossentropy) for multi-class classification
- [Focal loss](https://leimao.github.io/blog/Focal-Loss-Explained/) used for unbalanced datasets
- [Weighted focal loss](https://github.com/andrijdavid/FocalLoss/blob/master/focalloss.py) for multilabel classification
- [Weighted kappa](https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits) for multiclass classification
- [BCE with logit loss](https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits) to get sigmoid cross-entropy
- Custom [mimic loss](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/discussion/103280) used in [Jigsaw unintended](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/overview) bias classification competition
- [MTL custom loss](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/discussion/101630) used in [jigsaw unintended](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/overview) bias classification competition

Optimizers

- [Stochastic gradient descent](https://towardsdatascience.com/stochastic-gradient-descent-clearly-explained-53d239905d31)
- [RMSprop](https://towardsdatascience.com/understanding-rmsprop-faster-neural-network-learning-62e116fcf29a)
- [Adagrad](https://medium.com/konvergen/an-introduction-to-adagrad-f130ae871827) allows the learning rate to adapt based on parameters
- [Adam](https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/) for fast and easy convergence
- [Adam with warmup](https://www.kaggle.com/httpwwwfszyc/bert-keras-with-warmup-and-excluding-wd-parameters/) to enable warmup state to Adam algorithm
- [Bert Adam](https://huggingface.co/transformers/migration.html#optimizers-bertadam-openaiadam-are-now-adamw-schedules-are-standard-pytorch-schedules) for Bert based models
- [Rectified Adam](https://arxiv.org/pdf/1908.03265.pdf) for stabilizing training and accelerating convergence

Callback methods

Callbacks are always useful to monitor the performance of your model while training and trigger some necessary actions that can enhance the performance of your model.

- [Model checkpoint](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint) for monitoring and saving weights
- [Learning rate scheduler](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/LearningRateScheduler) to change the learning rate based on model performance to help converge easily
- Simple custom callbacks using [lambda callbacks](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/LambdaCallback)
- [Custom Checkpointing](https://machinelearningmastery.com/check-point-deep-learning-models-keras/)
- Building your [custom callbacks](https://keunwoochoi.wordpress.com/2016/07/16/keras-callbacks/) for various use cases
- [Reduce on plateau](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ReduceLROnPlateau) to reduce the learning rate when a metric has stopped improving
- [Early Stopping](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping) to stop training when the model stops improving
- [Snapshot ensembling](https://machinelearningmastery.com/snapshot-ensemble-deep-learning-neural-network/) to get a variety of model checkpoints in one training
- [Fast geometric ensembling](https://arxiv.org/abs/1802.10026)
- [Stochastic Weight Averaging (SWA)](https://www.kaggle.com/c/google-quest-challenge/discussion/119371)
- [Dynamic learning rate decay](https://arxiv.org/abs/1905.05583)

Evaluation and cross-validation

Choosing a suitable validation strategy is very important to avoid huge shake-ups or poor performance of the model in the private test set.

The traditional 80:20 split wouldn&#39;t work for many cases. Cross-validation works in most cases over the traditional single train-validation split to estimate the model performance.

There are different variations of KFold cross-validation such as group k-fold that should be chosen accordingly.

- [K-fold cross-validation](https://machinelearningmastery.com/k-fold-cross-validation/)
- [Stratified KFold cross-validation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html)
- [Group KFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html)
- [Adversarial validation](https://www.kaggle.com/konradb/adversarial-validation-and-other-scary-terms) to check if train and test distributions are similar or not
- [CV analysis of different strategies](https://www.kaggle.com/ratthachat/quest-cv-analysis-on-different-splitting-methods/)

Runtime tricks

You can perform some tricks to decrease the runtime and also improve model performance at the runtime.

- [Sequence bucketing](https://www.kaggle.com/bminixhofer/speed-up-your-rnn-with-sequence-bucketing/) to save runtime and improve performance
- [Get sentences from its head and tail](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/discussion/97443) when the input sentence is larger than 512 tokens
- [Use the GPU efficiently](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/discussion/89498)
- [Free keras memory](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/discussion/96876)
- [Save and load models](https://machinelearningmastery.com/save-load-keras-deep-learning-models/) to save runtime and memory
- [Don&#39;t Save Embedding in RNN Solutions](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/discussion/93230)
- Load [word2vec vectors](https://www.kaggle.com/c/quora-insincere-questions-classification/discussion/77968) without key vectors

Model ensembling

If you&#39;re in the competing environment one won&#39;t get to the top of the leaderboard without ensembling. Selecting the appropriate ensembling/stacking method is very important to get the maximum performance out of your models.

Let&#39;s see some of the popular ensembling techniques used in Kaggle competitions:

- [Weighted average ensemble](https://machinelearningmastery.com/weighted-average-ensemble-for-deep-learning-neural-networks/)
- [Stacked generalization ensemble](https://machinelearningmastery.com/stacking-ensemble-for-deep-learning-neural-networks/)
- [Out of folds predictions](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/52224)
- [Blending with linear regression](https://www.kaggle.com/suicaokhoailang/blending-with-linear-regression-0-688-lb)
- Use [optuna](https://github.com/pfnet/optuna) to determine blending weights
- [Power average ensemble](https://medium.com/data-design/reaching-the-depths-of-power-geometric-ensembling-when-targeting-the-auc-metric-2f356ea3250e)
- [Power 3.5 blending strategy](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/discussion/100661)
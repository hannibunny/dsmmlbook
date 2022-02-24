<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#Evolution-of-Textmodelling-and--classification" data-toc-modified-id="Evolution-of-Textmodelling-and--classification-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Evolution of Textmodelling and -classification</a></span></li><li><span><a href="#Word-Embeddings" data-toc-modified-id="Word-Embeddings-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Word Embeddings</a></span><ul class="toc-item"><li><span><a href="#Representations-of-single-Words" data-toc-modified-id="Representations-of-single-Words-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Representations of single Words</a></span></li><li><span><a href="#Representations-of-Documents" data-toc-modified-id="Representations-of-Documents-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Representations of Documents</a></span><ul class="toc-item"><li><span><a href="#Bag-of-Word-model" data-toc-modified-id="Bag-of-Word-model-2.2.1"><span class="toc-item-num">2.2.1&nbsp;&nbsp;</span>Bag-of-Word model</a></span></li><li><span><a href="#Sequences-of-Word-Vectors" data-toc-modified-id="Sequences-of-Word-Vectors-2.2.2"><span class="toc-item-num">2.2.2&nbsp;&nbsp;</span>Sequences of Word-Vectors</a></span></li><li><span><a href="#Vector-Representations-of-Sentences,-Paragraphs-and-Documents" data-toc-modified-id="Vector-Representations-of-Sentences,-Paragraphs-and-Documents-2.2.3"><span class="toc-item-num">2.2.3&nbsp;&nbsp;</span>Vector-Representations of Sentences, Paragraphs and Documents</a></span></li></ul></li><li><span><a href="#CBOW--and-Skipgram--Wordembedding" data-toc-modified-id="CBOW--and-Skipgram--Wordembedding-2.3"><span class="toc-item-num">2.3&nbsp;&nbsp;</span>CBOW- and Skipgram- Wordembedding</a></span><ul class="toc-item"><li><span><a href="#Continous-Bag-Of-Words-(CBOW)" data-toc-modified-id="Continous-Bag-Of-Words-(CBOW)-2.3.1"><span class="toc-item-num">2.3.1&nbsp;&nbsp;</span>Continous Bag-Of-Words (CBOW)</a></span></li><li><span><a href="#Skip-Gram" data-toc-modified-id="Skip-Gram-2.3.2"><span class="toc-item-num">2.3.2&nbsp;&nbsp;</span>Skip-Gram</a></span></li></ul></li><li><span><a href="#Other-Word-Embeddings" data-toc-modified-id="Other-Word-Embeddings-2.4"><span class="toc-item-num">2.4&nbsp;&nbsp;</span>Other Word-Embeddings</a></span><ul class="toc-item"><li><span><a href="#Integration-of-Fasttext-Word-Embeddings" data-toc-modified-id="Integration-of-Fasttext-Word-Embeddings-2.4.1"><span class="toc-item-num">2.4.1&nbsp;&nbsp;</span>Integration of Fasttext Word-Embeddings</a></span></li><li><span><a href="#Glove-Word-Embeddings" data-toc-modified-id="Glove-Word-Embeddings-2.4.2"><span class="toc-item-num">2.4.2&nbsp;&nbsp;</span>Glove Word-Embeddings</a></span></li></ul></li><li><span><a href="#Examples-of-Semantic-Relatedness-from-Word-Embeddings" data-toc-modified-id="Examples-of-Semantic-Relatedness-from-Word-Embeddings-2.5"><span class="toc-item-num">2.5&nbsp;&nbsp;</span>Examples of Semantic Relatedness from Word-Embeddings</a></span><ul class="toc-item"><li><span><a href="#Visualisation-through-t-SNE-projection-in-2-dim-space" data-toc-modified-id="Visualisation-through-t-SNE-projection-in-2-dim-space-2.5.1"><span class="toc-item-num">2.5.1&nbsp;&nbsp;</span>Visualisation through t-SNE projection in 2-dim space</a></span></li><li><span><a href="#Find-most-similar-words" data-toc-modified-id="Find-most-similar-words-2.5.2"><span class="toc-item-num">2.5.2&nbsp;&nbsp;</span>Find most similar words</a></span></li><li><span><a href="#Vector-representations-of-semantic-relations" data-toc-modified-id="Vector-representations-of-semantic-relations-2.5.3"><span class="toc-item-num">2.5.3&nbsp;&nbsp;</span>Vector representations of semantic relations</a></span></li><li><span><a href="#Lexical-Contrast-Injection" data-toc-modified-id="Lexical-Contrast-Injection-2.5.4"><span class="toc-item-num">2.5.4&nbsp;&nbsp;</span>Lexical Contrast Injection</a></span></li><li><span><a href="#Multilingual-Embeddings" data-toc-modified-id="Multilingual-Embeddings-2.5.5"><span class="toc-item-num">2.5.5&nbsp;&nbsp;</span>Multilingual Embeddings</a></span></li></ul></li></ul></li></ul></div>

[Go to Workshop Overview (.ipynb)](Overview.ipynb)

# Evolution of Textmodelling and -classification
![NLP Overall Picture](./Pics/overAllPicture.png)

# Word Embeddings
* Author: Johannes Maucher
* Last Update: 31.07.2018


## Representations of single Words
There are different ways to represent single words at the input of a machine learning algorithm. The conventional (old) way is to apply a **one-hot-encoding**. In this approach each word is represented by a vector $\mathbf{v_w}$, whose length is given by the number of words in the applied vocabulary $V$. Each word $w \in V$ is uniquely mapped to an integer $i_w \in [0,|V|]$ and the one-hot encoded word-vector $\mathbf{v_w}$ contains a 1 at position $i_w$ and zeros at all other positions. Drawbacks of the one-hot representation of words are:
* the one-hot encoded vectors are very long and sparse
* semantic or syntactic relations between words, can not be infered from the corresponding one-hot encoded vectors. 

The better way of representing words is to apply a **word-embedding**. Word embeddings have revolutionalized many fields of Natural Language Processing since their efficient neural-network-based generation has been published in [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/pdf/1301.3781.pdf) by Mikolov et al (2013). Word embeddings map words into vector-spaces such that semantically or syntactically related words are close together, whereas unrelated words are far from each other. Moreover, it has been shown that the word-embeddings, generated by *word2vec*-techniques *CBOW* or *Skipgram*, are well-structured in the sense that also relations such as *is-capital-of*, *is-female-of*, *is-plural-of* are encoded in the vector space. In this way questions like *woman is to queen, as man is to ?* can be answered by simple operations of linear algebra in the word-vector-space. Compared to the length of one-hot encoded word-vectors, word-embedding-vectors are short (typical lengths in the range from 100-300) and dense (float-values). 

![DSM concept](./Pics/dsm.png)

*CBOW* and *Skipgram*, are techniques to learn word-embeddings, i.e. a mapping of words to vectors by relatively simple neural networks. Usually large corpora are applied for learning, e.g. the entire Wikipedia corpus in a given language. Today, pretrained models for the most common languages are available, for example from this [Facebook github repository](https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md?utm_campaign=buffer&utm_content=buffer0df9b&utm_medium=social&utm_source=linkedin.com).

For the classifier implemented in [notebook K04germanNewsFeedClassification.ipynb](K04germanNewsFeedClassification.ipynb), a self-trained german word-embedding is applied. The embedding has been trained with CBOW from the dump of the entire german Wikipedia in notebook [K03generateCBOWfromWiki.ipynb](K03generateCBOWfromWiki.ipynb)


## Representations of Documents

### Bag-of-Word model
The conventional way of modelling documents in tasks like information-retrieval, document-clustering, document-classification, sentiment-analysis, topic-classification is to represent each document as a **Bag-Of-Word**-vector $$\mathbf{d}_i=(tf_{i,0},tf_{i,1},\ldots tf_{i,|V|})$$. Each component of this vector corresponds to a single term $j$ of the underlying vocabulary $V$ and the values $tf_{i,j}$ counts the frequency of term $j$ in document $i$. Instead of the term-frequency $tf_{i,j}$ it is also possible to fill the BoW-vector with 
* a binary indicator which indicates if the term $j$ appears in document $i$
* the tf-idf-values $$tfidf_{i,j}=tf_{i,j} \cdot log \frac{N}{df_j},$$ where $df_j$ is the frequency of documents, in which term $j$ appears, and $N$ is the total number of documents.

Independent of the values used, the BoW-model has the following major drawbacks:
* the order by which terms appear in the document is totally ignored
* semantic relatedness of terms is not modelled
* BoW- vectors are very long and sparse

### Sequences of Word-Vectors
Using word-embeddings all of the mentioned drawbacks of BoW-modells can be avoided. One approach is to represent the sequence of words in the document as a sequence of word-embedding vectors. These sequences can then be passed to e.g. Convolutional Neural Networks (CNN) or Long-Short-Term-Memory Networks (LSTM).  

![BoWvsEmbedding](./Pics/bowVsEmbedding.png)

### Vector-Representations of Sentences, Paragraphs and Documents
Word-Embeddings are based on the idea, that semantically related words can be mapped to vectors, which are close together. Where two words are semantically related, if they frequently appear in the same context (neighbouring words). This idea can also be applied to sentences or word-sequences in general. By applying e.g. LSTMs, Doc2Vec, Encoder-Decoder architectures etc. vector representations of texts can be generated, such that semantically related texts have similiar vector-representations.

## CBOW- and Skipgram- Wordembedding
In 2013 Mikolov et al. published [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/pdf/1301.3781.pdf). They proposed quite simple neural network architectures to efficiently create word-embeddings: CBOW and Skipgram. These architectures are better known as **Word2Vec**. In both techniques neural networks are trained for a pseudo-task. After training, the network itself is usually not of interest. However, the learned weights in the input-layer constitute the word-embeddings, which can then be applied for a large field of NLP-tasks, e.g. document classification.

### Continous Bag-Of-Words (CBOW)
The idea of CBOW is to predict the target word $w_i$, given the $N$ context-words $w_{i-N/2},\ldots, w_{i-1}, \quad w_{i+1}, w_{i+N/2}$. 
In order to learn such a predictor a large but unlabeled corpus is required. The extraction of training-samples from a corpus is sketched in the picture below:

![cbowTrainSamples](./Pics/cbowTrainSamples.png)

In this example a context length of $N=4$ has been applied. The first training-element consists of 
* the $N=4$ input-words *(happy,families,all,alike)*
* the target word *are*.

In order to obtain the second training-sample the window of length $N+1$ is just shifted by one to the right. The concrete architecture for CBOW is shown in the picture below. At the input the $N$ context words are one-hot-encoded. The fully-connected *Projection-layer* maps the context words to a vector representation of the context. This vector representation is the input of a softmax-output-layer. The output-layer has as much neurons as there are words in the vocabulary $V$. Each neuron uniquely corresponds to a word of the vocabulary and outputs an estimation of the probaility, that the word appears as target for the current context-words at the input.  
![cbowArchitecture](./Pics/cbowGramArchitecture.png)

Once the CBOW-network is trained, the vector representation of word $w$ are the weights from the one-hot encoded word $w$ at the input of the network to the neurons in the projection-layer. I.e. the number of neurons in the projection layer define the length of the word-embedding.

### Skip-Gram
Skip-Gram is similar to CBOW, but has a reversed prediction process: For a given target word at the input, the Skip-Gram model predicts words, which are likely in the context of this target word. Again, the context is defined by the $N$ neighbouring words. The extraction of training-samples from a corpus is sketched in the picture below:

![skipGramTrainSamples](./Pics/skipGramTrainSamples.png)

Again a context length of $N=4$ has been applied. The first training-element consists of 
* the first target word *(happy)* as input to the network 
* the first context word *(families)* as network-output.

The concrete architecture for Skip-gram is shown in the picture below. At the input the target-word is one-hot-encoded. The fully-connected *Projection-layer* outputs the current vector representation of the target-word. This vector representation is the input of a softmax-output-layer. The output-layer has as much neurons as there are words in the vocabulary $V$. Each neurons uniquely corresponds to a word of the vocabulary and outputs an estimation of the probaility, that the word appears in the context of the current target-word at the input.  
![cbowArchitecture](./Pics/skipGramArchitecture.png)

## Other Word-Embeddings
CBOW- and Skip-Gram are possibly the most popular word-embeddings. However, there are more count-based and prediction-based methods to generate them, e.g. Random-Indexing, [Glove](https://nlp.stanford.edu/projects/glove/), [FastText](https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md).

### Integration of Fasttext Word-Embeddings
After downloading word embeddings from [FastText](https://fasttext.cc/docs/en/english-vectors.html) they can be imported as follows:

from gensim.models import KeyedVectors
# Creating the model
pathMacBookAir='/Users/johannesmaucher/DataSets/fasttextEnglish300.vec'
pathMacBook='/Users/maucher/DataSets/Gensim/FastText/fasttextEnglish300.vec'
pathDeepLearn='../../DataSets/FastText/fasttextEnglish300.vec'
w2vmodel = KeyedVectors.load_word2vec_format(pathMacBook)
EMBEDDING_DIM=w2vmodel.vector_size

# Printing out number of tokens available
print("Number of Tokens: {}".format(len(w2vmodel.index2word)))
# Printing out the dimension of a word vector 
print("Dimension of a word vector: {}".format(EMBEDDING_DIM))

w="than"
print("First 10 components of word-vector of word {}: \n".format(w),w2vmodel[w][:10])

### Glove Word-Embeddings
After downloading word-embeddings from [Glove](https://nlp.stanford.edu/projects/glove/), they can be imported as follows:

import os
import numpy as np
w2vmodel={}
GLOVE_DIR = "/Users/maucher/DataSets/glove.6B"
#GLOVE_DIR = "./Data/glove.6B"
#embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    w2vmodel[word]=coefs
    #embeddings_index[word] = coefs
f.close()
EMBEDDING_DIM=100
print('Total %s word vectors in Glove 6B 100d.' % len(w2vmodel))

## Examples of Semantic Relatedness from Word-Embeddings
### Visualisation through t-SNE projection in 2-dim space

![SemanticRelatednessVis](./Pics/semanticRelatednessVis.png)

Source: [http://metaoptimize.s3.amazonaws.com/cw-embeddings-ACL2010/embeddings-mostcommon.EMBEDDING_SIZE=50.png](http://metaoptimize.s3.amazonaws.com/cw-embeddings-ACL2010/embeddings-mostcommon.EMBEDDING_SIZE=50.png)

### Find most similar words
![collobertSimilarities](./Pics/collobertSimilarities.png).

Source: [R. Collobert, NLP (almost) from Scratch](https://arxiv.org/pdf/1103.0398v1.pdf)

### Vector representations of semantic relations
![semanticRelations](./Pics/semanticRelations.png)

Source: [T. Mikolov et al, Linguistic Regularities in Continuous Space Word Representations](https://www.aclweb.org/anthology/N/N13/N13-1090.pdf)

### Lexical Contrast Injection
![lexicalContrast](./Pics/lexicalContrastInjection.png)
Source: [Pham et al, A Multitask Objective to Inject Lexical Contrast into Distributional Semantics](https://www.aclweb.org/anthology/P15-2004)


### Multilingual Embeddings
![Bilingual Embedding](./Pics/bilingualEmbedding.png)
Source: [Ruder et al, A Survey of Cross-lingual Word Embedding Models](https://arxiv.org/pdf/1706.04902.pdf)


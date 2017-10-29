# HMM-LDA-CPP
"Integrating Topics and Syntax" T.L. Griffiths et al. C++ implementation of HMM-LDA. Inspired by katsuya94/hmm-lda

The code is not tested on 20-newsgroups Topic modeling dataset and work as expecte but only in certain specific setting. Ideally is should work correctly with proper hyperparameter tuning but that takes too much time. Following hack reduces the chances of code to find undesirable output.

- Run the model in "topics only" setting with #classes = 3 (topic_class, sentence_start, sentence_end) and desired #topics for specific number of iterations until you get desired topic assignments.
- Use that topic assignments to initialize the full model with more #classes and same #topics. This ensures that some words already have a strong bias towards being topic words irrespective of random class assignments.


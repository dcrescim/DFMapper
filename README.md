DFMapper
========

This module is heavily influenced by the awesome pandas-sklearn module, as well as the Pipeline class from scikit-learn.

More often than not, one has to perform the same transformations on the training set, and the test set leading to a duplication of code, and a source of errors. This gets more complicated if one has to determine the number of categorical variables (say), and use that mapping on the test set. 

This module aims to make that whole process easier. By creating a DFMapper object, one can use the Transformer API to map both the training dataframe and the test dataframe, which makes the code much easier to understand and much more maintainable. 

Here are some example uses.


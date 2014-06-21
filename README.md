DFMapper
========

This module is heavily influenced by the awesome pandas-sklearn module, as well as the Pipeline class from scikit-learn.

More often than not, one has to perform the same transformations on the training set, and the test set leading to a duplication of code, and a source of errors. 

This module aims to make that whole process easier. By creating a DFMapper object, one can use the Transformer API to map both the training dataframe and the test dataframe, which makes the code much easier to understand and much more maintainable. 

Here are some example uses.

```python
df_train = pd.read_table('train.csv', sep=',')
df_test = pd.read_table('test.csv', sep=',')

mapper = DFMapper()
mapper.add_x('Pclass', LabelBinarizer())
mapper.add_x('Sex', [lambda x: x == 'male', LabelBinarizer()])
mapper.add_Y('Survived')

X_train, Y_train = mapper.fit_transform(df_train)
X_test, _ = mapper.transform(df_test)
```

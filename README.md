DFMapper
========

This module is heavily influenced by the awesome pandas-sklearn module, as well as the Pipeline class from scikit-learn.

More often than not, one has to perform the same transformations on the training set, and the test set leading to a duplication of code, and a source of errors. 

This module aims to make that whole process easier. By creating a DFMapper object, one can use the Transformer API to map both the training dataframe and the test dataframe, which makes the code much easier to understand and much more maintainable. 

Here is an example use using the intro Titanic Dataset from Kaggle.

```python
import pandas as pd
from DFMapper import *
from sklearn.preprocessing import LabelBinarizer

df_train = pd.read_table('train.csv', sep=',')
df_test = pd.read_table('test.csv', sep=',')

mapper = DFMapper()
mapper.add_X('Pclass', LabelBinarizer())
mapper.add_X('Sex', [lambda x: x == 'male', LabelBinarizer()])
mapper.add_Y('Survived')

X_train, Y_train = mapper.fit_transform(df_train)
X_test, _ = mapper.transform(df_test)
```

We see from the above example that we can register both x variables and y variables from a given dataframe. Moreover, we can apply the transformation stored in the mapping layer to the test set (ignoring, of course, the result. If we had the answers at hand these challenges would be a lot easier).

Also see, that we can specify a simple list instead of a Pipeline object, and interject our own functions into the list. The line

```python
mapper.add_X('Sex', [lambda x: x == 'male', LabelBinarizer()])
```

first passes that column of our dataframe through our lambda function, mapping it into a column of True/Falses, and then we split that through sklearn`s LabelBinarizer, to get 0 and 1's.


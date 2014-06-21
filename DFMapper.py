import numpy as np
import ipdb
import itertools

def isinstance_func(x):
  return hasattr(x, '__call__')

# Takes numpy array, and returns a row array
def row(arr):
  if len(arr.shape) == 1:
    return arr.reshape(1, len(arr))
  return arr

def col(arr):
  if len(arr.shape) == 1:
    return arr.reshape(len(arr), 1)
  return arr


def explode(matrix, order):
  cols = matrix.shape[1]
  assert order > 1, "order is not greater than 1"

  new_cols =  []
  for combos in itertools.combinations(xrange(cols),order):
    first_column_index = combos[0]

    # Create the combination column
    combo_column = np.copy(matrix[:,first_column_index])
    for cur_column_index in combos[1:]:
      combo_column *= matrix[:, cur_column_index]


    new_cols.append(col(combo_column))

  return np.hstack(new_cols)


class DFMapper(object):
  def __init__(self):
    self.dict_list = []
    self.index = None
    self.options = {}
  # Key is a column of the original data
  # function list is a list of one of the following
  #   - A class that implements the Transformer API
  #   - A function
  def _add(self, key, function_list, is_X, is_Y, is_index, as_col=True):
    
    if not isinstance(function_list, list):
      function_list = [function_list]

    if isinstance(key, str):
      key = [key]

    dict_values = {}
    dict_values['pipeline'] = function_list
    dict_values['is_X'] = is_X
    dict_values['is_Y'] = is_Y
    dict_values['is_index'] = is_index
    dict_values['as_col'] = as_col

    self.dict_list.append((key,dict_values))
  
  def add_X(self, key, function_list=[], as_col = True):
    self._add(key, function_list, is_X=True, is_Y=False, is_index=False, as_col=as_col)

  def add_Y(self, key, function_list=[], as_col = True):
    self._add(key, function_list, is_X=False, is_Y=True, is_index=False, as_col=as_col)

  def add_index(self, key, function_list=[], as_col=True):
    self._add(key, function_list, is_X=False, is_Y=False, is_index=True, as_col=as_col)

  def add_option(self, key, val=True):
    self.options[key] = val

  def evaluate(self, key, dict_options, df, eval_type):
    for el in key:
      if (el not in df):
        # If you are missing an X column, this is bad. 
        #   You should find it.
        if dict_options['is_X']:
          ValueError("The column %s is not in your dataframe" % key)
        
        # If you are missing Y columns, that is not a big deal
        #   You could just be transforming the test set.
        if dict_options['is_Y']:
          return None

    if dict_options['as_col']:
      cur_val = col(df[key].values)
    else:
      cur_val = df[key]

    #import ipdb; ipdb.set_trace()
    for (index, f) in enumerate(dict_options['pipeline']):
      if isinstance_func(f):
        cur_val = f(cur_val)
      else:
        if 'fit_transform' == eval_type:
          cur_val = f.fit_transform(cur_val)
        elif 'transform' == eval_type:
          cur_val = f.transform(cur_val)
        elif 'fit' == eval_type:
          # Just call fit at the end
          # otherwise call fit transform
          if index+1 == len(dict_options['pipeline']):
            f.fit(cur_val)
            return None
          else:
            cur_val = f.fit_transform(cur_val)
        else:
          assert False, "Only support options fit, transform and fit_transform"

    return cur_val

  def eval_and_coalesce(self, df, eval_type):
    results_X = []
    results_Y = []
    for (key, dict_options) in self.dict_list:
      cur_val = self.evaluate(key,dict_options, df, eval_type)

      # This occurs when you are trying to evaluate
      # a key that is not in the dataframe
      if cur_val == None:
        continue

      if dict_options['is_X']:
        results_X.append(cur_val)
      if dict_options['is_Y']:
        results_Y.append(cur_val)
      if dict_options['is_index']:
        self.index = cur_val

    results_X = np.hstack(results_X) if results_X else np.array([])
    results_Y = np.hstack(results_Y) if results_Y else np.array([])

    if ('explode' in self.options) and (len(results_X) > 0):
      order = self.options['explode']
      results_X = np.hstack([results_X, explode(results_X,order)])
    return results_X, results_Y

  def fit(self, df):
    self.eval_and_coalesce(df, 'fit')
    return self

  def transform(self, df):
    return self.eval_and_coalesce(df, 'transform')

  def fit_transform(self, df):
    return self.eval_and_coalesce(df, 'fit_transform')


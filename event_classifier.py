# event_classifier.py
# version 0.2.1
# History:
# 0.2.2 (2020/02/12): revamped and split up functions
# 0.2.1 (2020/02/11): Added documentation, some error checking
# 0.2.0 (2020/02/11): Reworked into class and methods rather than script
# 0.1.0 (2020/02/06): Initial script hacked together.

import numpy as np
import pandas as pd
import astropy.table as table
import tensorflow as tf
from sklearn.model_selection import train_test_split

#uf_columns = ['time','crsv','crsu','amp_sf','av1','av2','av3','au1','au2','au3','rawx','rawy','chipx','chipy','tdetx','tdety','detx','dety','x','y','pha','pi','sumamps','chip_id','status']
#cl_columns = ['time','tg_r','tg_d','chipx','chipy','tdetx','tdety','detx','dety','x','y','chip_id','pha','pi','tg_m','tg_lam','tg_mlam','tg_srcid','tg_part','tg_smap','status']

METRICS = [
  tf.keras.metrics.TruePositives(name='tp'),
  tf.keras.metrics.TrueNegatives(name='tn'),
  tf.keras.metrics.FalsePositives(name='fp'),
  tf.keras.metrics.FalseNegatives(name='fn'),
  tf.keras.metrics.BinaryAccuracy(name='binary_accuracy'),
  tf.keras.metrics.Precision(name='precision'),
  tf.keras.metrics.Recall(name='recall'),
  tf.keras.metrics.AUC(name='auc'),
]

class EventClassifier():
  """ Classifier for Chandra event data 

  Parameters
  ----------
  uf_file: string
    unfiltered (good+bad events) event file (default evt0)
  cl_file: string
    filtered (only good events) event file (default evt2)
  uf_columns: list of strings, optional
    list of columns to keep from uf_file. Applied _before_ processing
  cl_column: list of strings, optional
    list of columns to keep from cl_file. Applied _after_ processing but before merging.
  status_name: string, optional
    name of "status" column (bitmask for event status)
  batch_size: integer, optional
    batch size for tensorflow datasets, default 32
  dense_layers: tuple of integers
    tuple of dense layer sizes, each becomes a tf.keras.layers.Dense(N) layer in the model (default (128,)) 
  load_and_initialize: bool
    run the process_events() and compile_model() steps on initialization

  Methods
  -------
  process_events(self,expand_status=True):
    Load and process the input FITS files (uf_file and cl_file) into a Pandas dataframe (self.dataFrame)
  compile_model(self,dense_layers=(32,),model=None):
    Define and compile the model, or compile a user-provided model
  fit(self,train_data=None,val_data=None,epochs=15):
    Train (fit) the model
  train_test_split(self,test_frac=0.2,val_frac=0.2):
    Split self.dataFrame into training, validation, and testing Tensorflow datasets  
  create_dataset(self,data=None,batch_size=32,shuffle=True):
    Create tensorflow dataset from input Pandas dataframe
  get_input_len(self):
    Helper function to get number of input columns in dataset
  """
  def __init__(self,
      uf_file='evt0',cl_file='evt2',
      uf_columns = ['time','pha','status'],cl_columns=['time','pha'],
      status_name='status',load_and_initialize=True,
      batch_size=1024,dense_layers=(32,)):
    self.batch_size = batch_size
    self.uf_file = uf_file
    self.cl_file = cl_file
    self.status_name = status_name
    self.uf_columns = uf_columns
    self.cl_columns = cl_columns
    self.uf_table = None
    self.cl_table = None
    if load_and_initialize:
      self.process_events(uf_file=uf_file,cl_file=cl_file,status_name=self.status_name)
      # compile_model(dense_layers=dense_layers)

# METHOD: load_fits
# Load a FITS file and expand the "status" field into individual columns
# TODO:
# - make status-expanding more universal: check for fields with array data and
#   expand all of them?
  def load_fits(self,fname,status_name='status',verbose=0):
    """ Load a FITS event list and expand the "status" field

    Parameters
    ----------
    fname: str
      File name of FITS eventlist to load
    status_name: str, optional
      Name of column containing bitmask for event status (default 'status')
    verbose: int, optional
      If nonzero, print some messages about loading and processing data
    """
    if verbose:
      print('Loading file {}'.format(fname))
    fits_file = table.Table.read(fname,1)

    if verbose:
      print('Converting status bits in {}'.format(file_name))
    num_status_bits = fits_file[status_name].shape[1]
    status_column_names = ['{}{:02}'.format(status_name,x) for x in np.arange(num_status_bits)]
    status_columns = [table.Column(col,name) for col,name 
        in zip(np.int16(fits_file[status_name].swapaxes(0,1)),status_column_names)]
    fits_file.add_columns(status_columns)
    fits_file.remove_column(status_name)
    return fits_file, status_column_names

# METHOD: load_events
# Load unfiltered and filtered event files and store them in self.uf_table and
# self.cl_table
  def load_events(self,uf_file,cl_file,status_name='status'):
    """ Load unfiltered and filtered event files

    Parameters
    ----------
    uf_file: str
      Name of unfiltered event file (contains good+bad events, typically 'evt0')
    cl_file: str
      Name of filtered event file (contains only good events, typically 'evt0')
    """
    for fname,cols,table_attr in zip([uf_file,cl_file],[self.uf_columns,self.cl_columns],['uf_table','cl_table']):
      target, status_cols = self.load_fits(fname,status_name=status_name)
      if status_name in cols:
        cols.remove(status_name)
        cols.extend(status_cols)
      target.keep_columns(cols)
      setattr(self,table_attr,target)

# METHOD: process_events
# Read in the unfiltered and filtered event files, expand the 'status' field,
# and convert to a Pandas dataframe.
# TODO: 
# - allow for appending more data rather than overwriting self.dataFrame
# - have option to use self.uf_file, etc in addition to user-provided filenames
  def process_events(self,uf_file='evt0',cl_file='evt2',status_name='status'):
    """ Load FITS files, merge, and process into a Pandas dataframe

    Reads the fits files specified in self.uf_file and self.cl_file. Expands
    each element of the 'status' field (an array of booleans) into its own
    column. Merges uf_file and cl_file into a single table, and converts that
    table into a Pandas dataframe, stored in self.dataFrame.

    Parameters
    ----------
    expand_status: bool
      Whether or not to expand the "status" field into a bunch of fields (since "status" is usually an array)
    """
    self.load_events(uf_file=uf_file,cl_file=cl_file,status_name=status_name)
    self.full_table = table.join(left=self.uf_table,right=self.cl_table,
        keys='time',table_names=['uf','cl'],join_type='left')
# Astropy Tables use MaskedColumn columns to indicate where there is missing data.
# In our joined table, rows with bad events (i.e., in uf_table but not in
# cl_table) will have the "pha_cl" field masked, which we can use as our
# "bad_event" column.
# TODO: check for whether this actually is a MaskedColumn
    self.full_table.add_column(table.Column(np.int16(self.full_table['pha_cl'].mask)),name='bad_event')

    self.full_table.rename_column(name='pha_uf',new_name='pha')
    self.full_table.remove_column('pha_cl')
    self.full_table.remove_column('time')

    self.dataFrame = self.full_table.to_pandas()

# METHOD: create_dataset
# Starting from a Pandas Dataframe, create a TensorFlow dataset
# TODO: is it better to just do all the fitting straight from the dataframes,
# rather than making a TF dataset?
  def create_dataset(self,data=None,batch_size=32,shuffle=True):
    """ Create a TensorFlow dataset out of a Pandas DataFrame

    Parameters
    ----------
    data: pandas.DataFrame
      DataFrame to build the TF dataset out of.
    batch_size: integer, optional
      Batch size for tensorflow dataset. Default 32.
    shuffle: boolean
      Randomly shuffle the dataset
    """
    data = data.copy()
    labels = data.pop('bad_event')
    ds = tf.data.Dataset.from_tensor_slices((data.values,labels.values))
    if shuffle:
      ds = ds.shuffle(buffer_size=len(data.values))
    ds = ds.batch(batch_size)
    return ds

# METHOD: get_input_len
# Return the number of columns of data to be run through the network
# This is len(self.dataFrame.columns)-1 because the 'bad_event' column will be the label
  def get_input_len(self):
    """ Get the size of the input array for the neural network
    """
    return len(self.dataFrame.columns)-1

# METHOD: compile_model
# Define and compile the model. If model is not given, generate a basic
# sequential dense model. Otherwise use the model provided by the user.
  def compile_model(self,dense_layers=(32,),model=None,optimizer='adam',
      loss='binary_crossentropy',metrics=METRICS):
    """ Define and compile the neural network model

    Parameters
    ----------
    dense_layers: tuple of integers
      number of neurons per layer for a basic sequential dense network
    model: tf.keras.Model instance
      User-defined Keras model. Overrides dense_layers.
    """
    if(model == None):
      self.model = tf.keras.models.Sequential()
      self.model.add(tf.keras.layers.Input(self.get_input_len()))
      for dl in dense_layers:
        self.model.add(tf.keras.layers.Dense(dl,activation='relu'))
      self.model.add(tf.keras.layers.Dense(1,activation='sigmoid'))
    self.model.compile(optimizer=optimizer,loss=loss,metrics=metrics)

# METHOD: train_test_split
# Just runs sklearn's train_test_split on self.dataFrame and converts the
# resulting train/val/test dataframes to TensorFlow datasets
  def train_test_split(self,test_frac=0.2,val_frac=0.2):
    """ Create training, validation, and testing TensorFlow datasets

    Splits self.dataFrame into training, testing, and validation DataFrames and
    then converts those to TensorFlow datasets.

    Parameters
    ----------
    test_frac: float (0 to 1)
      fraction of entire dataset to be used for the "testing" dataset
    val_frac: float (0 to 1)
      fraction of _training_ dataset to be used for validation
    """
    self.train,self.test = train_test_split(self.dataFrame,test_size=test_frac)
    self.train,self.val = train_test_split(self.train,test_size=val_frac)
    self.train_data = self.create_dataset(self.train,shuffle=True)
    self.val_data = self.create_dataset(self.val,shuffle=False)
    self.test_data = self.create_dataset(self.test,shuffle=False)

# METHOD: fit
# Train the model. If datasets haven't been defined, generate them first.
# TODO:
# - check if the model actually exists
  def fit(self,train_data=None,val_data=None,epochs=15):
    """ Train (fit) the model

    Parameters
    ----------
    train_data: tensorflow.data.Dataset, optional
      Training data. If not provided, self.train_test_split is run with default parameters.
    val_data: tensorflow.data.Dataset, optional
      Validation data. If train_data is not provided, this is ignored.
    """
    if(train_data == None and not(hasattr(self,"train_data"))):
      self.train_test_split(test_frac=0.2,val_frac=0.2)
      train_data = self.train_data
      val_data = self.val_data
    self.model.fit(train_data,validation_data=val_data,epochs=epochs)


# TEST CODE
uf_file = 'data/evt0'
cl_file = 'data/evt2'
ec = EventClassifier(uf_file=uf_file,cl_file=cl_file,
  dense_layers=(32,),load_and_initialize=False,batch_size=2048)
#ec.fit(epochs=1)

# Finally, get the predictions for the test data and print out some basic statistics
# predictions = ec.model.predict(ec.test_data)
# predictions = predictions.reshape(predictions.shape[0])
# targets = test['bad_event'].values
# hist = np.histogram(targets + predictions)
# 
# print('Quick-and-dirty statistics: Add the predictions to the target values.')
# print('Since we either have 1 or 0, close to 0 or 2 is good')
# print('(because 1+1 = 2, 0+0 = 0), while close to 1 is bad.')
# print('---------------------------------------------')
# for value,binlo,binhi in zip(hist[0],hist[1][:-1],hist[1][1:]):
#   print('{:3.1f}-{:3.1f}: {:>8}'.format(binlo,binhi,value))

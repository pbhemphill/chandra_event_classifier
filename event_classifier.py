import numpy as np
import pandas as pd
import astropy.table as table
import tensorflow as tf
from sklearn.model_selection import train_test_split

#uf_columns = ['time','crsv','crsu','amp_sf','av1','av2','av3','au1','au2','au3','rawx','rawy','chipx','chipy','tdetx','tdety','detx','dety','x','y','pha','pi','sumamps','chip_id','status']
#cl_columns = ['time','tg_r','tg_d','chipx','chipy','tdetx','tdety','detx','dety','x','y','chip_id','pha','pi','tg_m','tg_lam','tg_mlam','tg_srcid','tg_part','tg_smap','status']

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
  def __init__(self,uf_file='evt0',cl_file='evt2',batch_size=32,
      dense_layers=(128,),status_name='status',load_and_initialize=True,
      uf_columns = ['time','pha','status'],cl_columns=['time','pha']):
    self.batch_size = batch_size
    self.uf_file = uf_file
    self.cl_file = cl_file
    self.status_column_name = status_name
    self.uf_columns = uf_columns
    self.cl_columns = cl_columns
    if load_and_initialize:
      self.process_events(expand_status=True)
      self.compile_model(dense_layers=dense_layers)

# METHOD: process_events
# Read in the unfiltered and filtered event files, and optionally expand the
# 'status' bit if it's there.
# Why are we expanding the "status" field? Astropy's Table is nice, but its
# to_pandas() method fails if there are field with arrays. This is messy, but
# it preserves the actual _information_ in the status bit much better!
# TODO: 
# - make status-expanding more universal: check for fields with array data and
#   expand all of them?
# - allow for appending more data rather than overwriting self.dataFrame
  def process_events(self,expand_status=True):
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
    fits_files = []
    for file_name in [self.uf_file,self.cl_file]:
      print('Loading file {}'.format(file_name))
      fits_file = table.Table.read(file_name,1)
      if file_name is self.uf_file:
        fits_file.keep_columns(self.uf_columns)
      if expand_status:
        print('Converting status bits in {}'.format(file_name))
        num_status_bits = fits_file[self.status_column_name].shape[1]
        status_column_names = ['{}{:02}'.format(self.status_column_name,x) for x in np.arange(num_status_bits)]
        status_columns = [table.Column(col,name) for col,name 
            in zip(np.int16(fits_file[self.status_column_name].swapaxes(0,1)),status_column_names)]
        fits_file.add_columns(status_columns)
        fits_file.remove_column(self.status_column_name)
      fits_files.append(fits_file)
    uf = fits_files[0]
    cl = fits_files[1]
# Remove extra columns from evt2 data
    fits_files[1].keep_columns(self.cl_columns)
    full_table = table.join(left=fits_files[0],right=fits_files[1],
        keys='time',table_names=['uf','cl'],join_type='left')
# Astropy Tables have "masked" columns to indicate where there is "bad" data.
# In our joined table, rows with bad events (i.e., in uf but not in cl) will
# have the "pha_cl" field masked, which we can convert to our "bad_event" column.
    full_table.add_column(table.Column(np.int16(full_table['pha_cl'].mask)),name='bad_event')
# Clean up the data table
    full_table.rename_column(name='pha_uf',new_name='pha')
    full_table.remove_column('pha_cl')
    full_table.remove_column('time')
# Finally, convert to a Pandas dataframe
    self.dataFrame = full_table.to_pandas()

# METHOD: create_dataset
# Starting from a Pandas Dataframe, create a TensorFlow dataset
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
  def compile_model(self,dense_layers=(32,),model=None):
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
    self.model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

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
    train,test = train_test_split(self.dataFrame,test_size=test_frac)
    train,val = train_test_split(train,test_size=val_frac)
    self.train_data = self.create_dataset(train,shuffle=True)
    self.val_data = self.create_dataset(val,shuffle=False)
    self.test_data = self.create_dataset(test,shuffle=False)

# METHOD: fit
# Train the model. If datasets haven't been defined, generate them first.
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
  dense_layers=(128,),load_and_initialize=True,batch_size=1024)
ec.fit(epochs=1)

# Finally, get the predictions for the test data and print out some basic statistics
predictions = ec.model.predict(ec.test_data)
predictions = predictions.reshape(predictions.shape[0])
targets = test['bad_event'].values
hist = np.histogram(targets + predictions)

print('Quick-and-dirty statistics: Add the predictions to the target values.')
print('Since we either have 1 or 0, close to 0 or 2 is good')
print('(because 1+1 = 2, 0+0 = 0), while close to 1 is bad.')
print('---------------------------------------------')
for value,binlo,binhi in zip(hist[0],hist[1][:-1],hist[1][1:]):
  print('{:3.1f}-{:3.1f}: {:>8}'.format(binlo,binhi,value))

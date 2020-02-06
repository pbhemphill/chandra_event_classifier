import numpy as np
import pandas as pd
from astropy.table.column import Column
from astropy.table import Table,join
import tensorflow as tf
from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

file_names = ['evt0','evt2']
fits_files = []
for file_name in file_names:
  print('Loading file {}'.format(file_name))
  fits_file = Table.read(file_name,1)
  print('Converting status bits in {}'.format(file_name))
  status_column_names = ['status{:02}'.format(x) for x in np.arange(32)]
  status_columns = [Column(col,name) for col,name 
      in zip(np.int16(fits_file['status'].swapaxes(0,1)),status_column_names)]
  fits_file.add_columns(status_columns)
  fits_file.remove_column('status')
  fits_files.append(fits_file)

uf = fits_files[0]
cl = fits_files[1]

# Remove columns that are only in the cleaned data (these are only set for good
# events, since only good events end up in the cleaned data, so we can't have
# them in the training data)
# cl.remove_columns(set(cl.columns) - set(uf.columns))
# Actually, remove everything but "time" (so we have something to match on) and
# "pha" (whose mask we will use to determine whether an event was good or bad).
cl.keep_columns(['time','pha'])

# Join the FITS data tables. Join on time since that's unique for each event.
full_table = join(left=uf,right=cl,keys='time',table_names=['uf','cl'],join_type='left')

# Now full_table has a 'pha_cl' column which is a MaskedColumn type - mask ==
# True means a row was not in the cleaned eventlist, and therefore was a bad event.
full_table.add_column(Column(np.int16(full_table['pha_cl'].mask)),name='bad_event')
# We'll also rename the 'pha_uf' column (which was the pha column from the
# unfiltered eventlist) to just 'pha'
full_table.rename_column(name='pha_uf',new_name='pha')
full_table.remove_column('pha_cl')
full_table.remove_column('time')

# ndx = np.arange(len(full_table))
# np.random.shuffle(ndx)
# ndx = ndx[0:100000]
# full_table = full_table[ndx]

full_df = full_table.to_pandas()
train, test = train_test_split(full_df,test_size=0.2)
train, val = train_test_split(train,test_size=0.2)
print(len(val), 'validation examples')
print(len(test), 'test examples')

# Setting up the tensorflow dataset
# For testing, only pick a (random) subset of the rows:
def create_dataset(dataframe,shuffle=True,batch_size=32,label_name='bad_event'):
  dataframe = dataframe.copy()
  labels = dataframe.pop(label_name)
  ds = tf.data.Dataset.from_tensor_slices((dataframe.values,labels.values))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe.values))
  ds = ds.batch(batch_size)
  return ds

batch_size=1024
train_data = create_dataset(train,batch_size=batch_size)
val_data = create_dataset(val,shuffle=False,batch_size=batch_size)
test_data = create_dataset(test,shuffle=False,batch_size=batch_size)

# feature_layer = layers.DenseFeatures(feature_columns)
# dense_layer1 = layers.Dense(128,activation='relu')
# output_layer = layers.Dense(1,activation='sigmoid')

model = tf.keras.Sequential([
  layers.Input(len(train.columns)-1),
  layers.Dense(128,activation='relu'),
  layers.Dense(1,activation='sigmoid')])

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
# model.summary()

model.fit(train_data,validation_data=val_data,epochs=15)

predictions = model.predict(test_data)
predictions = predictions.reshape(predictions.shape[0])
targets = test['bad_event'].values
hist = np.histogram(targets + predictions)

print('Quick-and-dirty statistics: Add the predictions to the target values.')
print('Since we either have 1 or 0, close to 0 or 2 is good')
print('(because 1+1 = 2, 0+0 = 0), while close to 1 is bad.')
print('---------------------------------------------')
for value,binlo,binhi in zip(hist[0],hist[1][:-1],hist[1][1:]):
  print('{:3.1f}-{:3.1f}: {:>8}'.format(binlo,binhi,value))

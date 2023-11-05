import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import re
import numpy as np
import pandas as pd
import time



stk = pd.read_csv('filtered.tsv', sep='\t')

print("Dataset length:", stk.shape[0])



#Remove rows where ref_tox or trn_tox lie between 0.03 and 0.97
#To leave sentences that shows the difference most clearly
#stk = stk[~((stk['ref_tox'] >= 0.03) & (stk['ref_tox'] <= 0.97) | (stk['trn_tox'] >= 0.03) & (stk['trn_tox'] <= 0.97))]

#Sort by similarity to start with those pairs of sentences whose differences are most easy to understand
stk = stk.sort_values(by='similarity', ascending=False)

print("Filtered dataset length:", stk.shape[0])



def preprocess_sentence(w):
  w = w.lower().strip()

  #Removing consecutive three dots, because they get separated and counted as three separate words
  #With greaty increases tensor dimension
  w = re.sub(r'\.{3,}', '', w)

  #Creating a space between a word and the punctuation following it
  w = re.sub(r"([.,!?])", r" \1 ", w)

  #Replacing everything with space, except letters, punctuations, etc.
  w = re.sub(r"[^a-zA-Z?.!,']+", " ", w)

  #Replacing multiple consecutive whitespaces with a single space
  w = re.sub('\s{2,}', ' ', w)

  w = w.strip()

  #Adding start and end tokens
  w = '<start> ' + w + ' <end>'
  return w
  
  
  
def create_dataset(df, num_examples):
  sentence_pairs = []

  for index, row in df.iterrows():
    if len(sentence_pairs) >= num_examples:
      break

    #Filtering out all sentences with more than 10 words
    if len(row['reference'].split(" ")) > 10 or len(row['translation'].split(" ")) > 10:
      continue

    ref = preprocess_sentence(row['reference'])
    tr = preprocess_sentence(row['translation'])

    #Filtering out all sentences that after preprocessing have more than 25 tokens
    if len(ref.split(" ")) > 25 or len(tr.split(" ")) > 25:
      print("Inadequate ref sentence skipped:", ref)
      print("Inadequate tr sentence skipped:", tr)
      continue

    if row['ref_tox'] < row['trn_tox']:
      sentence_pairs.append([ref, tr])
    else:
      sentence_pairs.append([tr, ref])

  return zip(*sentence_pairs)

#View sample of the dataset
tr, ref = create_dataset(stk, 30000)
print(ref[:20])
print(tr[:20])



#Tokenize the sentence and pad the sequence to the same length
def tokenize(lang):
  lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
  lang_tokenizer.fit_on_texts(lang)
  tensor = lang_tokenizer.texts_to_sequences(lang)
  tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')
  return tensor, lang_tokenizer
  
  
  
def load_dataset(df, num_examples=None):
  targ_lang, inp_lang = create_dataset(df, num_examples)
  input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
  target_tensor, targ_lang_tokenizer = tokenize(targ_lang)
  return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer
  
  
  
num_examples = 300000
input_tensor, target_tensor, inp_lang, targ_lang = load_dataset(stk, num_examples)

#Calculate max_length of the target tensors
max_length_targ, max_length_inp = target_tensor.shape[1], input_tensor.shape[1]
print("Max length of a row in tensors:")
print(max_length_targ, max_length_inp)



#Creating training and validation sets using an 80-20 split
input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.2)

#Show length
print("Splitted tensors sizes:")
print(len(input_tensor_train), len(target_tensor_train), len(input_tensor_val), len(target_tensor_val))
print(input_tensor_train[0])
print(target_tensor_train[0])



#Configuration
BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = 64
steps_per_epoch = len(input_tensor_train) // BATCH_SIZE #Number of batches per epoch
steps_per_epoch_val = len(input_tensor_val) // BATCH_SIZE #Number of batches per epoch
embedding_dim = 256 #For word embeddings
units = 1024 #Dimensionality of the output space of the RNN
vocab_inp_size = len(inp_lang.word_index) + 1
vocab_tar_size = len(targ_lang.word_index) + 1

dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
validation_dataset = tf.data.Dataset.from_tensor_slices((input_tensor_val, target_tensor_val)).shuffle(BUFFER_SIZE)
validation_dataset = validation_dataset.batch(BATCH_SIZE, drop_remainder=True)

example_input_batch, example_target_batch  = next(iter(dataset))
print("Input and target batches sizes")
print(example_input_batch.shape, example_target_batch.shape)



class Encoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
    super(Encoder, self).__init__()
    self.batch_sz = batch_sz
    self.enc_units = enc_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.enc_units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')

  def call(self, x, hidden):
    #x shape: (batch, sequence_len)
    #hidden shape: (batch, units)
    x = self.embedding(x)
    #x after embedding shape: (batch, sequence_size, embedding_dim)

    output, state = self.gru(x, initial_state=hidden)
    #output shape: (batch, sequence_len, units)
    #state shape: (batch, units)
    return output, state

  def initialize_hidden_state(self):
    return tf.zeros((self.batch_sz, self.enc_units))
    
    
    
encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)

#Sample input
sample_hidden = encoder.initialize_hidden_state()
sample_output, sample_hidden = encoder(example_input_batch, sample_hidden)
print ('Encoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))
print ('Encoder Hidden state shape: (batch size, units) {}'.format(sample_hidden.shape))



class Decoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
    super(Decoder, self).__init__()
    self.batch_sz = batch_sz
    self.dec_units = dec_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.dec_units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
    self.fc = tf.keras.layers.Dense(vocab_size)

  def call(self, x, hidden):
    #x shape: (batch, 1)
    #hidden shape: (batch, units)
    x = self.embedding(x)
    #x after embedding shape: (batch, 1, embedding_dim)

    output, state = self.gru(x, initial_state=hidden)
    #output shape: (batch, 1, units)
    #state shape: (batch, units)

    output = tf.reshape(output, (-1,output.shape[2]))
    #output shape: (batch, units)

    x = self.fc(output)
    #x shape: (batch, vocab_size)

    return x, state
    
    
    
class DotProductAttention(tf.keras.layers.Layer):
  def __init__(self,units):
    super(DotProductAttention,self).__init__()
    self.WK = tf.keras.layers.Dense(units)
    self.WQ= tf.keras.layers.Dense(units)

  def call(self, query, values):
    #query shape: (batch, units)
    #values shape: (batch, sequence_len, units)

    #Adding dimension for matrix multiplication
    query_with_time_axis = tf.expand_dims(query,1)
    #query_with_time_axis shape: (batch, 1, units)

    K = self.WK(values)
    #K shape: (batch, sequence_len, units)
    #Dimension didn't change, because dense layer size == units

    Q = self.WQ(query_with_time_axis)
    #Q shape: (batch, 1, units)
    #Same here

    #Transposing matrix for multiplication that follows
    QT = tf.einsum('ijk->ikj',Q)
    #QT shape: (batch, units, 1)

    score = tf.matmul(K, QT)
    #score shape: (batch, sequence_len, 1)

    attention_weights = tf.nn.softmax(score,axis=1)
    #attention_weights shape: (batch, sequence_len, 1)

    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)
    #context_vector shape: (batch, units)

    return context_vector, attention_weights
    
    
    
attention_layer = DotProductAttention(units)
attention_result, attention_weights = attention_layer(sample_hidden, sample_output)

print("Attention result shape: (batch size, units) {}".format(attention_result.shape))
print("Attention weights shape: (batch_size, sequence_length, 1) {}".format(attention_weights.shape))



class DecoderWithAttention(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz, attention=None):
    super(DecoderWithAttention, self).__init__()
    self.batch_sz = batch_sz
    self.dec_units = dec_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.dec_units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
    self.fc = tf.keras.layers.Dense(vocab_size)
    self.attention = attention

  def call(self, x, hidden, enc_output):
    #x shape: (batch, 1)
    #hidden shape: (batch, units)
    #output shape: (batch, sequence_len, units)

    x = self.embedding(x)
    attention_weights = None
    #x after embedding shape: (batch, 1, embedding_dim)

    if self.attention:
      context_vector, attention_weights = self.attention(hidden, enc_output)
      #context_vector shape: (batch, units)
      #context_vector after expansion shape: (batch, 1, units)

      #x: (batch, 1, embedding_dim)
      x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    #x after attention shape: (batch, 1, units + embedding_dim)

    output, state = self.gru(x, initial_state=hidden)
    #output shape: (batch, 1, units)
    #state shape: (batch, units)

    output = tf.reshape(output, (-1,output.shape[2]))
    #output shape: (batch, units)

    x = self.fc(output)
    #x shape: (batch, vocab_size)

    return x, state, attention_weights
    
    
    
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def loss_function(real, pred):
  loss_ = loss_object(real, pred)
  return tf.reduce_mean(loss_)
  
  
  
print(loss_object([1,2], [[0,0.6,0.3,0.1],[0,0.6,0.3,0.1]]))
print(loss_function([1,2], [[0,0.6,0.3,0.1],[0,0.6,0.3,0.1]]))



optimizer = tf.keras.optimizers.Adam()

def get_train_step_function():

  @tf.function
  def train_step(inp, targ, enc_hidden, encoder, decoder):
    loss = 0

    with tf.GradientTape() as tape:
      #Sending input and initialized hidden state to encoder
      enc_output, enc_hidden = encoder(inp, enc_hidden)

      dec_hidden = enc_hidden

      #Sending 64 start tokens at the very beginning
      dec_input = tf.expand_dims([targ_lang.word_index['<start>']]*BATCH_SIZE, 1)

      for t in range(1, targ.shape[1]):
        predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

        #predictions are logits, targ[:, t] is index in vocab
        loss += loss_function(targ[:, t], predictions)

        dec_input = tf.expand_dims(targ[:,t],1)

    batch_loss = (loss/int(targ.shape[1]))
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss
  return train_step
  
  
  
def calculate_validation_loss(inp, targ, enc_hidden, encoder, decoder):
  loss = 0
  enc_output, enc_hidden = encoder(inp, enc_hidden)
  dec_hidden = enc_hidden

  #Sending 64 start tokens at the very beginning
  dec_input = tf.expand_dims([targ_lang.word_index['<start>']]*BATCH_SIZE,1)

  for t in range(1, targ.shape[1]):
    predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

    #predictions are logits, targ[:, t] is index in vocab
    loss+=loss_function(targ[:,t], predictions)
    dec_input = tf.expand_dims(targ[:, t], 1)

  loss = loss/int(targ.shape[1])
  return loss
  
  
  
def training_seq2seq(epochs,attention):
  encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
  decoder = DecoderWithAttention(vocab_tar_size, embedding_dim, units, BATCH_SIZE, attention)
  train_step_func = get_train_step_function()
  training_loss = []
  validation_loss = []

  for epoch in range(epochs):
    start = time.time()
    enc_hidden = encoder.initialize_hidden_state()
    total_loss = 0

    #Training
    for (batch,(inp,targ)) in enumerate(dataset.take(steps_per_epoch)): #Enumerating all batches
      batch_loss = train_step_func(inp, targ, enc_hidden, encoder, decoder)
      total_loss += batch_loss

      if batch%100 == 0:
        print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, batch, batch_loss))

    enc_hidden = encoder.initialize_hidden_state()
    total_val_loss = 0

    #Validation
    for (batch, (inp, targ)) in enumerate(validation_dataset.take(steps_per_epoch_val)):
      val_loss = calculate_validation_loss(inp, targ, enc_hidden, encoder, decoder)
      total_val_loss += val_loss

    training_loss.append(total_loss/steps_per_epoch)
    validation_loss.append(total_val_loss/steps_per_epoch_val)

    print('Epoch {} Loss {:.4f} Validation Loss {:.4f}'.format(epoch + 1, training_loss[-1], validation_loss[-1]))
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

  return encoder, decoder, training_loss, validation_loss
  
  
  
epochs = 3
attention = DotProductAttention(units)
print("Running seq2seq model with dot product attention")
encoder_dp, decoder_dp, training_loss, validation_loss = training_seq2seq(epochs, attention)

tloss = training_loss
vloss = validation_loss



encoder_dp.save_weights('models/encoder_dp')
decoder_dp.save_weights('models/decoder_dp')

import pickle

with open('models/inp_lang.pkl', 'wb') as outp:
    pickle.dump(inp_lang, outp, pickle.HIGHEST_PROTOCOL)

with open('models/targ_lang.pkl', 'wb') as outp:
    pickle.dump(targ_lang, outp, pickle.HIGHEST_PROTOCOL)
    
with open('models/parameters.pkl', 'wb') as outp:
    tup = (max_length_targ, max_length_inp, BUFFER_SIZE, BATCH_SIZE, steps_per_epoch, steps_per_epoch_val, embedding_dim, units, vocab_inp_size, vocab_tar_size)
    pickle.dump(tup, outp, pickle.HIGHEST_PROTOCOL)
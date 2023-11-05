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






import pickle

with open('models/inp_lang.pkl', 'rb') as inp:
    inp_lang = pickle.load(inp)
   
with open('models/targ_lang.pkl', 'rb') as inp:
    targ_lang = pickle.load(inp)
    
with open('models/parameters.pkl', 'rb') as inp:
    max_length_targ, max_length_inp, BUFFER_SIZE, BATCH_SIZE, steps_per_epoch, steps_per_epoch_val, embedding_dim, units, vocab_inp_size, vocab_tar_size = pickle.load(inp)
    
    

encoder_dp = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
attention = DotProductAttention(units)
decoder_dp = DecoderWithAttention(vocab_tar_size, embedding_dim, units, BATCH_SIZE, attention)

encoder_dp.load_weights('models/encoder_dp')
decoder_dp.load_weights('models/decoder_dp')




    
    
def translate(sentence, encoder, decoder):
  attention_plot = np.zeros((max_length_targ, max_length_inp))
  sentence = preprocess_sentence(sentence)

  #Manually tokenizing the input
  inputs = [inp_lang.word_index[i] for i in sentence.split(' ')]
  inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=max_length_inp, padding='post')

  inputs = tf.convert_to_tensor(inputs)

  result = ''

  #Sending input and initialized hidden state to encoder
  hidden = [tf.zeros((1, units))]
  enc_out, enc_hidden = encoder(inputs, hidden)

  dec_hidden = enc_hidden

  dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)

  #Sending each token to decoder and decoding the predicted logits
  for t in range(max_length_targ):
    predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out)
    predicted_id = tf.argmax(predictions[0]).numpy()
    result += targ_lang.index_word[predicted_id] + ' '

    if targ_lang.index_word[predicted_id] == '<end>':
      return result, sentence

    dec_input = tf.expand_dims([predicted_id], 0)
  return result, sentence
  
  
  
result, sentence = translate("fuck you", encoder_dp, decoder_dp)
print('Input: %s' % (sentence))
print('Predicted translation: {}'.format(result))

result, sentence = translate("get the hell outta here", encoder_dp, decoder_dp)
print('Input: %s' % (sentence))
print('Predicted translation: {}'.format(result))

result, sentence = translate("This house is fucking creepy", encoder_dp, decoder_dp)
print('Input: %s' % (sentence))
print('Predicted translation: {}'.format(result))

result, sentence = translate("I am mad at you", encoder_dp, decoder_dp)
print('Input: %s' % (sentence))
print('Predicted translation: {}'.format(result))

result, sentence = translate("Don't be so fucking rude", encoder_dp, decoder_dp)
print('Input: %s' % (sentence))
print('Predicted translation: {}'.format(result))

result, sentence = translate("I am so fucking pissed at your bullshit", encoder_dp, decoder_dp)
print('Input: %s' % (sentence))
print('Predicted translation: {}'.format(result))

result, sentence = translate("You're a stunning cunt", encoder_dp, decoder_dp)
print('Input: %s' % (sentence))
print('Predicted translation: {}'.format(result))

result, sentence = translate("I'm a pussy on a sofa", encoder_dp, decoder_dp)
print('Input: %s' % (sentence))
print('Predicted translation: {}'.format(result))
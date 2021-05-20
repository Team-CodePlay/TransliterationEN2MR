

import tensorflow as tf
print(tf.__version__)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import os
import io
import time
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import unicodedata
import re
import time
from pprint import pprint

print("imported all libs")

marathi_alphabets = [chr(alpha) for alpha in range(2304, 2432)]
marathi_alphabet_size = len(marathi_alphabets)

marathi_alpha2index = {'<start>': 0,'<end>' : 1}
for index, alpha in enumerate(marathi_alphabets):
    marathi_alpha2index[alpha] = index+1

# pprint(marathi_alpha2index)

def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s))
        # if unicodedata.category(c) != 'Mn')


def preprocess_sentence_english(w):
    w = unicode_to_ascii(w.lower().strip())
    w = w.replace('-', ' ').replace(',', ' ')
    
    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ." 
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    
    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
    
    w = w.rstrip().strip()
    
    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    w = '@' + w + '#'
    return w.split()

def preprocess_sentence_marathi(w):
    w = unicode_to_ascii(w.strip())
    w = w.replace('-', ' ').replace(',', ' ')
    cleaned_line = ''
    for char in w:
        if char in marathi_alpha2index or char == ' ':
            cleaned_line += char
    cleaned_line = cleaned_line.rstrip().strip()
    
    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    cleaned_line = '@' + cleaned_line + '#'
    return cleaned_line.split()

def create_dataset(data):
    lang1_words = []
    lang2_words = []

    for index, row in data.iterrows():
        try:
            wordlist1 = preprocess_sentence_english(row['en']) # clean English words.
            wordlist2 = preprocess_sentence_marathi(row['mr']) # clean marathi words.
            if len(wordlist1) != len(wordlist2):
                print('Skipping: ', row['en'], ' - ', row['mr'])
                continue
            for word in wordlist1:
                lang1_words.append(word)
            for word in wordlist2:
                lang2_words.append(word)
        except Exception as e:
            print(f"Found error at {row}")
    return [lang1_words,lang2_words]

train = pd.read_csv("model_files/mr.translit.sampled.train.tsv", sep="\t", names = ['mr', 'en', 'lex'], header=None)

train.dropna(inplace=True)

train_data = create_dataset(train)

print('dataset created')

class WordIndex():
  def __init__(self, lang):
    self.lang = lang
    self.word2idx = {}
    self.idx2word = {}
    self.vocab = set()
    
    self.create_index()
    
  def create_index(self):
    for phrase in self.lang:
      for l in phrase:
        self.vocab.update(l)
    
    self.vocab = sorted(self.vocab)
    
    self.word2idx['<pad>'] = 0
    for index, word in enumerate(self.vocab):
      self.word2idx[word] = index + 1
    
    for word, index in self.word2idx.items():
      self.idx2word[index] = word

def max_length(tensor):
    return max(len(t) for t in tensor)


def load_dataset(pairs, max_length_inp=None, max_length_tar=None):
    # index language using the class defined above    
    inp_lang = WordIndex(pairs[0])
    targ_lang = WordIndex(pairs[1])
    
    # Vectorize the input and target languages
    # English words
    input_tensor = [[inp_lang.word2idx[s] for s in en] for en in pairs[0]]
    
    # marathi words
    target_tensor = [[targ_lang.word2idx[s] for s in mr] for mr in pairs[1]]
    
    # Calculate max_length of input and output tensor
    # Here, we'll set those to the longest sentence in the dataset
    if max_length_inp is None or max_length_tar is None:
        max_length_inp, max_length_tar = max_length(input_tensor), max_length(target_tensor)
    
    # Padding the input and output tensor to the maximum length
    input_tensor = tf.keras.preprocessing.sequence.pad_sequences(input_tensor, 
                                                                 maxlen=max_length_inp,
                                                                 padding='post')
    
    target_tensor = tf.keras.preprocessing.sequence.pad_sequences(target_tensor, 
                                                                  maxlen=max_length_tar, 
                                                                  padding='post')
    
    return input_tensor, target_tensor, inp_lang, targ_lang, max_length_inp, max_length_tar

def convert(lang, tensor):
  for t in tensor:
    if t!=0:
      print ("%d ----> %s" % (t, lang.idx2word[t]))

print('loading dataset')
input_tensor, target_tensor, inp_lang, targ_lang, max_length_inp, max_length_tar = load_dataset(train_data)

print('loaded dataset')
input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.05)
print('dataset split')

len(input_tensor_train), len(target_tensor_train), len(input_tensor_val), len(target_tensor_val)

BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = 128
steps_per_epoch = len(input_tensor_train)//BATCH_SIZE
embedding_dim = 256
units = 1024
vocab_inp_size = len(inp_lang.word2idx)+1
vocab_tar_size = len(targ_lang.word2idx)+1

dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

example_input_batch, example_target_batch = next(iter(dataset))
example_input_batch.shape, example_target_batch.shape

class Encoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
    super(Encoder, self).__init__()
    self.batch_sz = batch_sz
    self.enc_units = enc_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.enc_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')

  def call(self, x, hidden):
    x = self.embedding(x)
    output, state = self.gru(x, initial_state = hidden)
    return output, state

  def initialize_hidden_state(self):
    return tf.zeros((self.batch_sz, self.enc_units))

encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)

# sample input
sample_hidden = encoder.initialize_hidden_state()
sample_output, sample_hidden = encoder(example_input_batch, sample_hidden)
print ('Encoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))
print ('Encoder Hidden state shape: (batch size, units) {}'.format(sample_hidden.shape))

class Attention(tf.keras.Model):
  def __init__(self, units):
    super(Attention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, query, values):
    # hidden shape == (batch_size, hidden size)
    # hidden_with_time_axis shape == (batch_size, 1, hidden size)
    # we are doing this to perform addition to calculate the score
    hidden_with_time_axis = tf.expand_dims(query, 1)

    # score shape == (batch_size, max_length, 1)
    # we get 1 at the last axis because we are applying score to self.V
    # the shape of the tensor before applying self.V is (batch_size, max_length, units)
    score = self.V(tf.nn.tanh(
        self.W1(values) + self.W2(hidden_with_time_axis)))

    # attention_weights shape == (batch_size, max_length, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights

attention_layer = Attention(10)
attention_result, attention_weights = attention_layer(sample_hidden, sample_output)

print("Attention result shape: (batch size, units) {}".format(attention_result.shape))
print("Attention weights shape: (batch_size, sequence_length, 1) {}".format(attention_weights.shape))

class Decoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
    super(Decoder, self).__init__()
    self.batch_sz = batch_sz
    self.dec_units = dec_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.dec_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.fc = tf.keras.layers.Dense(vocab_size)

    # used for attention
    self.attention = Attention(self.dec_units)

  def call(self, x, hidden, enc_output):
    # enc_output shape == (batch_size, max_length, hidden_size)
    context_vector, attention_weights = self.attention(hidden, enc_output)


    # x shape after passing through embedding == (batch_size, 1, embedding_dim)
    x = self.embedding(x)

    # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    # passing the concatenated vector to the GRU
    output, state = self.gru(x)

    # output shape == (batch_size * 1, hidden_size)
    output = tf.reshape(output, (-1, output.shape[2]))

    # output shape == (batch_size, vocab)
    x = self.fc(output)

    return x, state, attention_weights

decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)

sample_decoder_output, _, _ = decoder(tf.random.uniform((128, 1)),
                                      sample_hidden, sample_output)

print ('Decoder output shape: (batch_size, vocab size) {}'.format(sample_decoder_output.shape))

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)

checkpoint_dir = 'model_files/checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


def evaluate(sentence):
    attention_plot = np.zeros((max_length_tar, max_length_inp))

    sentence = preprocess_sentence_english(sentence)[0]

    inputs = [inp_lang.word2idx[i] for i in sentence]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                           maxlen=max_length_inp,
                                                           padding='post')
    inputs = tf.convert_to_tensor(inputs)
    result = ''
    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden = encoder(inputs, hidden)
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang.word2idx['@']], 0)

    options = []
    for t in range(max_length_tar):
        predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                             dec_hidden,
                                                             enc_out)

        # storing the attention weights to plot later on
        attention_weights = tf.reshape(attention_weights, (-1, ))
        attention_plot[t] = attention_weights.numpy()

        options_for_curr_char = []
        for idx, i in enumerate(np.argsort(predictions[0])[::-1][:2]):
            if idx >0 and predictions[0][i] > 10:
                options_for_curr_char.append({"letter": targ_lang.idx2word[i], "conf":  predictions[0][i].numpy()})
            elif idx==0:
                options_for_curr_char.append({"letter": targ_lang.idx2word[i], "conf":  predictions[0][i].numpy()})
            # print(f"{targ_lang.idx2word[i]} ({predictions[0][i]})", end = " ")
        options.append(options_for_curr_char)
        # print()
        predicted_id = tf.argmax(predictions[0]).numpy()
        result += targ_lang.idx2word[predicted_id] + ' '
        if targ_lang.idx2word[predicted_id] == '#':
            return options, result, sentence, attention_plot

        # the predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)

    return options, result, sentence, attention_plot

def plot_attention(attention, sentence, predicted_sentence):
#     print(predicted_sentence)
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')

    fontdict = {'fontsize': 14}

    ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()

def transliterate(sentence):
    options, result, sentence, attention_plot = evaluate(sentence)
    # for i in options:

    options_sorted = []
    print(options)
    def construct_options(s, temp, avg = 0):
        first = temp[0]

        for i in first:
            if len(temp[1:]) > 0:
                avg = max(0, avg)
                if i['letter'] == '#':
                    options_sorted.append(((avg + i['conf']), s))
                elif i['letter'] == 'ा':
                    construct_options(s, temp[1:], avg + i['conf'])
                    construct_options(s+ i['letter'], temp[1:], avg + i['conf'])
                else:
                    construct_options(s + i['letter'], temp[1:], avg + i['conf'])
            elif i['letter'] == '#':
                options_sorted.append(((avg + i['conf']), s))
            else:
                options_sorted.append(((avg + i['conf']), s + i['letter']))
    
    construct_options('', options)
    # print(options_sorted)
                
    # print(options)

    # print('Input: %s' % (sentence))
    # print('Predicted translation: {}'.format(''.join(result.split(' '))))
    # result = unicode_to_ascii(result)    
    # attention_plot = attention_plot[:len(result.split(' ')), :len(list(sentence))]
    # plot_attention(attention_plot, list(sentence), result.split(' '))
    return [i[1] for i in sorted(options_sorted, reverse=True)]

def transliterate_sentence(sentence):
    translated = []
    for i in sentence.split(" "):
        translated.append(transliterate(i)[:-1])
    return ' '.join(translated)

# transliterate_sentence("maala don dole aahet")

# transliterate("dole")
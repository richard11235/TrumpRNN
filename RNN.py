import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, MaxPooling2D, ReLU, GRU, Softmax, Embedding, Layer
#from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.layers import UpSampling2D, Conv2D, LSTM
from tensorflow.keras import Sequential, Model
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
import numpy as np

BATCH_SIZE = 32
SEQ_LENGTH = 140

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)
   
def build_model(vocab_size,batch_size=BATCH_SIZE ,verbose=False):
    model = Sequential([
    Embedding(vocab_size,256,batch_input_shape=[batch_size,None]),
    GRU(256,
        return_sequences=True,
        stateful=True,
        recurrent_initializer='glorot_uniform'),
    GRU(256,
        return_sequences=True,
        stateful=True,
        recurrent_initializer='glorot_uniform'),
    GRU(256,
        return_sequences=True,
        stateful=True,
        recurrent_initializer='glorot_uniform'),
    GRU(256,
        return_sequences=True,
        stateful=True,
        recurrent_initializer='glorot_uniform'),
    Dense(vocab_size)])
    #model.add(Softmax())
    if verbose:
        model.summary()
    return model

def load_data(txtfile='data/content.txt'):
    f = open(txtfile,'r',encoding='utf-8')
    lines = f.read()#.lower()
    f.close()
    lines = [line + '\n' for line in lines.split('\n') if is_ascii(line)]
    return ''.join(lines)

def is_ascii(s):
    return all(ord(c) < 256 for c in s)

def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


if __name__ == '__main__':
    fname = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
    lines = load_data()
    text = lines
    # length of text is the number of characters in it
    print ('Length of text: {} characters'.format(len(text)))
    vocab = sorted(set(text))
    print ('{} unique characters'.format(len(vocab)))

    char2idx = {u:i for i, u in enumerate(vocab)}
    idx2char = np.array(vocab)
    text_as_int = np.array([char2idx[c] for c in text])

    vocab_size = len(vocab)

    char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

    sequences = char_dataset.batch(SEQ_LENGTH + 1, drop_remainder=True)

    def split_input_target(chunk):
        input_text = chunk[:-1]
        target_text = chunk[1:]
        return input_text, target_text

    dataset = sequences.map(split_input_target)

    dataset = dataset.shuffle(10000).batch(BATCH_SIZE, drop_remainder=True)

    #print(dataset)

    
    model = build_model(vocab_size,verbose=True)

    optimizer = Adam(learning_rate=.001)

    model.compile(loss=loss,optimizer=optimizer)
    
    checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath='chkpt/ckpt_{epoch}',
    save_weights_only=True)

    try:
        model.load_weights(tf.train.latest_checkpoint('./chkpt'))
    except:
        pass

    history = model.fit(dataset, epochs=10, callbacks=[checkpoint_callback])




    
    def generate_text(model, start_string):
        # Evaluation step (generating text using the learned model)

        # Number of characters to generate
        num_generate = 1000

        # Converting our start string to numbers (vectorizing)
        input_eval = [char2idx[s] for s in start_string]
        input_eval = tf.expand_dims(input_eval, 0)

        # Empty string to store our results
        text_generated = []

        # Low temperatures results in more predictable text.
        # Higher temperatures results in more surprising text.
        # Experiment to find the best setting.
        temperature = 0.5

        # Here batch size == 1
        model.reset_states()
        for i in range(num_generate):
            predictions = model(input_eval)
            # remove the batch dimension
            predictions = tf.squeeze(predictions, 0)

            # using a categorical distribution to predict the character returned by the model
            predictions = predictions / temperature
            predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

            # We pass the predicted character as the next input to the model
            # along with the previous hidden state
            input_eval = tf.expand_dims([predicted_id], 0)

            text_generated.append(idx2char[predicted_id])

        return (start_string + ''.join(text_generated))

    model = build_model(vocab_size,batch_size=1)

    model.load_weights(tf.train.latest_checkpoint('./chkpt'))

    model.build(tf.TensorShape([1, None]))

    model.summary()

    print(generate_text(model, start_string=u"\n"))

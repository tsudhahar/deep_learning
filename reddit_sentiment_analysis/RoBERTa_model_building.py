import pandas as pd
# Recommended tensorflow version is <= 2.1.0, otherwise F1 score function breaks
import tensorflow as tf
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import tensorflow_datasets as tfds
from transformers import TFRobertaForSequenceClassification
from transformers import RobertaTokenizer
import os


# Load your Dataset
train_tweets = pd.read_csv('train_tweets.csv')
test_tweets = pd.read_csv('test_tweets.csv')

test_tweets['label'] = 0

training_sentences, testing_sentences = train_test_split(train_tweets[['tweet', 'label']],
                                                         test_size=0.2)

roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# can be up to 512 for BERT
max_length = 100

# the recommended batches size for BERT are 32,64 ... however on this dataset we are overfitting quite fast
# and smaller batches work like a regularization.
# You might play with adding another dropout layer instead.

batch_size = 64

def convert_example_to_feature(review):
    # combine step for tokenization, WordPiece vector mapping and will
    # add also special tokens and truncate reviews longer than our max length
    
    return roberta_tokenizer.encode_plus(review,
                                 add_special_tokens=True,  # add [CLS], [SEP]
                                 max_length=max_length,  # max length of the text that can go to RoBERTa
                                 pad_to_max_length=True,  # add [PAD] tokens at the end of sentence
                                 return_attention_mask=True,  # add attention mask to not focus on pad tokens
                                 )
    '''
    return roberta_tokenizer.encode_plus(review,
                                 add_special_tokens=True,  # add [CLS], [SEP]
                                 max_length=max_length,  # max length of the text that can go to RoBERTa
                                 padding=True,  # add [PAD] tokens at the end of sentence
                                 return_attention_mask=True,  # add attention mask to not focus on pad tokens
                                 )
    '''

# map to the expected input to TFRobertaForSequenceClassification, see here
def map_example_to_dict(input_ids, attention_masks, label):
    return {
      "input_ids": input_ids,
      "attention_mask": attention_masks,
           }, label

def encode_examples(ds, limit=-1):
    # Prepare Input list
    input_ids_list = []
    attention_mask_list = []
    label_list = []

    if (limit > 0):
        ds = ds.take(limit)

    for review, label in tfds.as_numpy(ds):
        bert_input = convert_example_to_feature(review.decode())
        input_ids_list.append(bert_input['input_ids'])
        attention_mask_list.append(bert_input['attention_mask'])
        label_list.append([label])

    return tf.data.Dataset.from_tensor_slices((input_ids_list,
                                               attention_mask_list,
                                               label_list)).map(map_example_to_dict)

training_sentences_modified = tf.data.Dataset.from_tensor_slices((training_sentences['tweet'],
                                                                  training_sentences['label']))

testing_sentences_modified = tf.data.Dataset.from_tensor_slices((testing_sentences['tweet'],
                                                                 testing_sentences['label']))

ds_train_encoded = encode_examples(training_sentences_modified).shuffle(10000).batch(batch_size)
ds_test_encoded = encode_examples(testing_sentences_modified).batch(batch_size)



learning_rate = 7e-5
number_of_epochs = 8

class ModelMetrics(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.count_n = 1

    def on_epoch_end(self, batch, logs={}):
        
        os.mkdir('reddit_model' + str(self.count_n))
        self.model.save_pretrained('reddit_model' + str(self.count_n)) # this folder address should match with folder we created above
        
        y_val_pred = tf.nn.softmax(self.model.predict(ds_test_encoded))
        y_pred_argmax = tf.math.argmax(y_val_pred, axis=1)
        testing_copy = testing_sentences.copy()
        testing_copy['predicted'] = y_pred_argmax
        f1_s = f1_score(testing_sentences['label'], testing_copy['predicted'])
        print('\n f1 score is :', f1_s)
        self.count_n += 1

metrics = ModelMetrics()

# model initialization
model = TFRobertaForSequenceClassification.from_pretrained("roberta-base")
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=1e-08)

# we do not have one-hot vectors, we can use sparce categorical cross entropy and accuracy
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
model.fit(ds_train_encoded, epochs=number_of_epochs,
          validation_data=ds_test_encoded, callbacks=[metrics])
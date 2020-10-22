import praw
import pandas as pd
from transformers import RobertaTokenizer
import tensorflow as tf
from transformers import TFRobertaForSequenceClassification
import tensorflow_datasets as tfds

reddit = praw.Reddit(client_id='c61aYydHGTRGkw',
                     client_secret='hEw2j00lh_w2inKcTAlcqq_kJT0',
                     user_agent='android:com.example.myredditapp:v1.2.3')

# get 10 hot posts from the MachineLearning subreddit
top_posts = reddit.subreddit('showerthoughts').top('week', limit=10)

max_length = 100
batch_size = 64
roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")


def convert_example_to_feature(review):
    # combine step for tokenization, WordPiece vector mapping and will
    # add also special tokens and truncate reviews longer than our max length
    return roberta_tokenizer.encode_plus(review,
                                         add_special_tokens=True,
                                         max_length=max_length,
                                         pad_to_max_length=True,
                                         return_attention_mask=True,
                                         )


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


def replies_of(top_level_comment, comment_list):
    if len(top_level_comment.replies) == 0:
        return
    else:
        for num, comment in enumerate(top_level_comment.replies):
            try:
                comment_list.append(str(comment.body))
            except:
                continue
            replies_of(comment, comment_list)


def main():
    # load saved model
    model = TFRobertaForSequenceClassification.from_pretrained('reddit_model5')

    list_of_subreddit = ['showerthoughts', 'askmen', 'askreddit', 'jokes', 'worldnews']
    for j in list_of_subreddit:
        # get 10 hot posts from the MachineLearning subreddit
        top_posts = reddit.subreddit(j).top('week', limit=10)
        comment_list = []
        # save subreddit comments in dataframe
        for submission in top_posts:
            submission_comm = reddit.submission(id=submission.id)

            for count, top_level_comment in enumerate(submission_comm.comments):
                try:
                    replies_of(top_level_comment, comment_list)
                except:
                    continue

        comment_dataframe = pd.DataFrame(comment_list, columns=['Comments'])
        comment_dataframe['label'] = 0
        print(comment_dataframe)

        # prepare data as per RoBERTa model input
        submission_sentences_modified = tf.data.Dataset.from_tensor_slices((comment_dataframe['Comments'],
                                                                            comment_dataframe['label']))
        ds_submission_encoded = encode_examples(submission_sentences_modified).batch(batch_size)

        # predict sentiment of Reddit comments
        submission_pre = tf.nn.softmax(model.predict(ds_submission_encoded))
        submission_pre_argmax = tf.math.argmax(submission_pre, axis=1)
        comment_dataframe['label'] = submission_pre_argmax

        negative_comments_count = comment_dataframe[comment_dataframe['label'] == 1].count()
        positive_comments_count = comment_dataframe[comment_dataframe['label'] == 0].count()

        print(f"overall sentiment of subreddit r/{j} are Positive comments: {positive_comments_count}"
              f" Negative comments: {negative_comments_count}")


if __name__ == '__main__':
    main()
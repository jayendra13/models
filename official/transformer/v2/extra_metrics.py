import tensorflow as tf

from official.transformer.v2 import extra_utils

def padded_accuracy(predicted_ids, labels):
  """Percentage of times that predictions matches labels on non-0s."""
  weights = tf.cast(tf.not_equal(labels, 0),tf.float32)

  labels = tf.cast(labels,tf.float32)
  m = tf.keras.metrics.Accuracy()
  _ = m.update_state(labels, predicted_ids, sample_weight=weights)
  return m.result().numpy()


def compute_accuracy(prediction_file, bleu_ref, subtokenizer, params):

  predicted_text = extra_utils.file_to_list(prediction_file)
  gt_text = extra_utils.file_to_list(bleu_ref)

  predicted_lines = [subtokenizer.encode(line) for line in predicted_text]
  gt_lines = [subtokenizer.encode(line) for line in gt_text]

  padded_predicted_lines = tf.keras.preprocessing.sequence.pad_sequences(
    predicted_lines,
    maxlen=params["decode_max_length"],
    dtype="int32",
    padding="post")

  padded_gt_lines = tf.keras.preprocessing.sequence.pad_sequences(
    gt_lines,
    maxlen=params["decode_max_length"],
    dtype="int32",
    padding="post")
  accuracy = padded_accuracy(padded_predicted_lines, padded_gt_lines)

  return accuracy


def parseable_percentage(prediction_file):
  predicted_text = extra_utils.file_to_list(prediction_file)
  total_predictions = len(predicted_text)

  count = 0
  for prediction in predicted_text:
    if extra_utils.is_parsable(prediction):
      count += 1

  return (count/total_predictions)*100

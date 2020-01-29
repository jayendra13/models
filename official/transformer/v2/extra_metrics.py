import tensorflow as tf

from official.transformer.v2 import extra_utils
from official.transformer.utils import metrics

def padded_accuracy(predicted_ids, labels):
  """Percentage of times that predictions matches labels on non-0s."""
  weights = tf.cast(tf.not_equal(labels, 0),tf.float32)

  labels = tf.cast(labels,tf.float32)
  m = tf.keras.metrics.Accuracy()
  _ = m.update_state(labels, predicted_ids, sample_weight=weights)
  return m.result().numpy()


def pad_list(lines, decode_max_length):
  padded_lines = tf.keras.preprocessing.sequence.pad_sequences(
    lines,
    maxlen=decode_max_length,
    dtype="int32",
    padding="post")
  return padded_lines


def encode_files_to_list(prediction_file, bleu_ref, subtokenizer):
  predicted_text = extra_utils.file_to_list(prediction_file)
  gt_text = extra_utils.file_to_list(bleu_ref)

  predicted_lines = [subtokenizer.encode(line) for line in predicted_text]
  gt_lines = [subtokenizer.encode(line) for line in gt_text]
  return predicted_lines, gt_lines


def convert_to_encoded_list(prediction_file, bleu_ref, subtokenizer, params):
  predicted_lines, gt_lines = encode_files_to_list(prediction_file, bleu_ref, subtokenizer)
  padded_predicted_lines = pad_list(predicted_lines, params["decode_max_length"])
  padded_gt_lines = pad_list(gt_lines, params["decode_max_length"])
  return padded_predicted_lines, padded_gt_lines


def compute_accuracy(prediction_file, bleu_ref, subtokenizer, params):

  padded_predicted_lines, padded_gt_lines = convert_to_encoded_list(prediction_file, bleu_ref, subtokenizer, params)
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


def bigram_rouge(prediction_file, bleu_ref, subtokenizer):
  predicted_lines, gt_lines = encode_files_to_list(prediction_file, bleu_ref, subtokenizer)
  return metrics.rouge_n(predicted_lines, gt_lines, 2)


def sentence_level_rouge(prediction_file, bleu_ref, subtokenizer):
  predicted_lines, gt_lines = encode_files_to_list(prediction_file, bleu_ref, subtokenizer)
  return metrics.rouge_l_sentence_level(predicted_lines, gt_lines)

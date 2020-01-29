import ast
import contextlib
import io
import tensorflow as tf


def process_text(code):

  code = code.split("\n")
  code = "<NEW_LINE>".join(code)

  code = code.split("\r")
  code = "<NEW_LINE>".join(code)

  code = code.split("\t")
  code = "<TAB>".join(code)

  code = code.split("    ")
  code = "<TAB>".join(code)

  return code


def reverse_preprocessing(txt):
  txt = txt.replace('<SOS> ','')
  new_txt = txt.replace('<NEW_LINE>','\n')
  new_txt = new_txt.replace('<TAB>','    ')

  lines = [line.strip() for line in new_txt.split("\n")]
  new_txt = "\n".join(lines)
  return new_txt


def file_to_list(fname):
  with open(fname, 'r') as f:
    lines = f.readlines()
  return lines


def is_parsable(text):
  code = reverse_preprocessing(text)
  try:
    ast.parse(code)
    return True
  except Exception as ex:
    return False


@contextlib.contextmanager
def tag(stream, name):
  stream.write("<%s>" % name)
  yield
  stream.write("</%s>" % name)


def data_generator(filenames):
  with contextlib.ExitStack() as stack:
    files = [stack.enter_context(open(fname)) for fname in filenames]
    data = [f.readlines() for f in files]
  for e in zip(*data):
    yield e


def build_table(source_file, ref_file, prediction_file):

  stream = io.StringIO()
  headers = ["Docstring", "Reference", "Prediction"]
  rows = data_generator([source_file, ref_file, prediction_file])

  count = 0
  with tag(stream, "table"):
    with tag(stream, "tr"):
      for header in headers:
        with tag(stream, "th"):
          stream.write(header)
    for row in rows:
      with tag(stream, "tr"):
        for entry in row:
          with tag(stream, "td"):
            with tag(stream, "pre"):
              stream.write(reverse_preprocessing(entry.strip()))

      count += 1
      if count > 1000:
        break
    s = stream.getvalue()
    # stream.close()
    return s


def log_predictions_to_tensorboard(logfile, source_file, ref_file, prediction_file):
  data = build_table(source_file, ref_file, prediction_file)
  with tf.summary.create_file_writer(logfile).as_default():
    tf.summary.text("table", data, 0)

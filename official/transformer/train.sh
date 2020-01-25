# Ensure that PYTHONPATH is correctly defined as described in
# https://github.com/tensorflow/models/tree/master/official#requirements
export PYTHONPATH=/home/jayendra/lab/models

cd /home/jayendra/lab/models/official/transformer/v2

# Export variables
PARAM_SET=big
DATA_DIR=/home/jayendra/lab/models/official/transformer/data
MODEL_DIR=$HOME/transformer/model_$PARAM_SET
VOCAB_FILE=$DATA_DIR/vocab.ende.32768

# Download training/evaluation/test datasets
#python3 data_download.py --data_dir=$DATA_DIR

# Train the model for 100000 steps and evaluate every 5000 steps on a single GPU.
# Each train step, takes 4096 tokens as a batch budget with 64 as sequence
# maximal length.
python3 transformer_main.py --data_dir=$DATA_DIR --model_dir=$MODEL_DIR \
    --vocab_file=$VOCAB_FILE --param_set=$PARAM_SET \
    --train_steps=100000 --steps_between_evals=5000 \
    --batch_size=4096 --max_length=64 \
    --bleu_source=$DATA_DIR/newstest2014.en \
    --bleu_ref=$DATA_DIR/newstest2014.de \
    --num_gpus=0 \
    --enable_time_history=false

# Run during training in a separate process to get continuous updates,
# or after training is complete.
tensorboard --logdir=$MODEL_DIR
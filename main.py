# main.py
import os
import sys
import args
import time
import random
import numpy as np
import tensorflow as tf

from data import Vocab
from batcher import Batcher
from model import GSNModel
from decode import BeamSearchDecoder
from train import train, evaluate
from utils import get_datapath, get_steps, set_random_seeds, make_hps

# Access flags for configuration
FLAGS = tf.app.flags.FLAGS

# Set the visible GPU devices based on the configuration
os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.device


def main(unused_argv):
    # Set random seeds for reproducibility
    set_random_seeds()

    # Configure dataset paths and steps based on data size
    get_datapath()
    get_steps()

    # Enable detailed logging
    tf.logging.set_verbosity(tf.logging.INFO)
    print(f"Now the mode of this run is {FLAGS.mode}!")

    # Create log directory if it doesn't exist
    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)

    # Modify configuration for decoding mode
    if FLAGS.mode == 'decode':
        FLAGS.branch_batch_size = FLAGS.beam_size  # Adjust batch size for beam search
        FLAGS.TS_mode = False  # Turn off teacher forcing

    # Create hyperparameters object
    hps = make_hps()

    # Initialize vocabulary
    vocab = Vocab(hps.vocab_path, hps.vocab_size)

    # Training or decoding
    if hps.mode == 'train':
        # Create batchers for training and evaluation
        batcher = Batcher(hps.data_path, vocab, hps)
        eval_hps = hps._replace(mode='eval')
        eval_batcher = Batcher(hps.eval_data_path, vocab, eval_hps)

        # Initialize the model and start training
        model = GSNModel(hps, vocab)
        train(model, batcher, eval_batcher, vocab, hps)
    elif hps.mode == 'decode':
        # Create batcher for test data
        decode_hps = hps._replace(max_dec_steps=1)
        batcher = Batcher(hps.test_data_path, vocab, decode_hps)

        # Initialize model and decoder, then start decoding
        model = GSNModel(decode_hps, vocab)
        decoder = BeamSearchDecoder(model, batcher, vocab)
        decoder._decode()


if __name__ == '__main__':
    tf.app.run()

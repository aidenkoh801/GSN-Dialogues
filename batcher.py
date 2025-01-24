# batcher.py
import glob
import time
import Queue
import struct
import numpy as np
import tensorflow as tf
from threading import Thread
from random import shuffle
from tensorflow.core.example import example_pb2
import json
import data
import pickle as pkl

FLAGS = tf.app.flags.FLAGS


class Batch:
    """
    Represents a batch of data for training or inference.
    """

    def __init__(self, tfexamples, hps, vocab, struct_dist):
        self.hps = hps

        # Initialize arrays for encoder input, attention masks, etc.

        """
        enc_batch stores the tokenized encoder input for each utterance in a batch.
        1st Dim: branch_batch_size; no. of dialogue branches in a batch
        2nd Dim: sen_batch_size; no. of utterances per dialogue branch. This will be padded or truncated.
        3rd Dim: max_enc_steps: max no. of tokens per utterances. This will be padded or truncated.

        branch_batch_size = no. of dialogue branches in a batch.
            if a batch contains 2 conversations, and one has 3 branches while the other has 2 branches, 
            the branch_batch_size would be 5 for this batch.

        sen_batch_size = no. of utterances per dialogue branch.
            number of utterances per dialogue branch can vary.
            To handle this variation, the code uses sen_batch_size to pad or truncate branches 
            so that all branches in the batch have a uniform number of utterances.

        max_enc_steps = max no. of tokens per utterance. 
            length of an utterance directly determines the number of tokens generated
            The utterance is truncated to include only the first max_enc_steps tokens.
            OR Padding is applied to shorter utterances to ensure that all utterances in enc_batch have the same length
        """
        self.enc_batch = np.zeros((hps.branch_batch_size, hps.sen_batch_size, hps.max_enc_steps), dtype=np.int32)
        """
        attn_mask ensures padded positions are ignored during attention mechanism.
            positions are set to a large negative value (e.g., -1e10),
            which effectively nullifies their contribution during the attention computation.
            Masked positions are reduced to a very large negative value, 
            so their probabilities become near-zero after applying the softmax function.
        """
        self.attn_mask = -1e10 * np.ones_like(self.enc_batch, dtype=np.float32)

        """
        dec_batch array is the input to the decoder during training. 
        Each row corresponds to a sequence of token IDs starting with a special <start> token.
        eg. if target_batch = [hello, how, are, you, <end>]
            then dec_batch is = [<start>, hello, how, are, you]
          branch_batch_size: 
            each dialogue branch in the batch corresponds to a unique context or conversation path.
            Since each branch requires its own response during training,
            dec_batch must have an entry for each dialogue branch in the batch.
          max_dec_steps:
            specifies the maximum number of tokens (words or subwords)
            that the decoder can process or generate for a single response.
        """
        self.dec_batch = np.zeros((hps.branch_batch_size, hps.max_dec_steps), dtype=np.int32)

        """
        target_batch: Represents the expected output (ground truth) for training.
          eg. If the target response is: [hello, how, are, you]
              Then the target batch is: [hello, how, are, you, <end>] OR [hello, how, are, you, <end>, <pad>, <pad>] if the response is short and needs padding.
        Contains the token IDs for the target response, ending with a <end> token.
        """
        self.target_batch = np.zeros_like(self.dec_batch)

        """
        padding_mark: A mask for the target sequence, indicating valid tokens versus padding.
        1 for valid tokens, 0 for padding positions.
        eg. If target batch = [hello, how, are, you, <end>, <pad>, <pad>]
            Then the padding mark is = [1, 1, 1, 1, 1, 0, 0]
        Helps calculate the loss only for valid tokens, ignoring padding.
        """
        self.padding_mark = np.zeros_like(self.dec_batch, dtype=np.float32)

        # State matrices, user relations, and embeddings
        """
        state_matrix: Represents graph-like relationships between utterances in a branch. 
        Connections between utterances in branch are based on: 
        1. if utterance is direct response to another utterance. (In the dataset, its relation_at)
        2. utterance from same speaker that follows a chronological order. (In the dataset, its relation_user)
        eg. Branch 1:
            [
              [1, 1, 1],  # Utterance 0 connects to itself, Utterances 1, and 2.
              [0, 1, 0],  # Utterance 1 connects to itself only.
              [0, 0, 1],  # Utterance 2 connects to itself only.
            ]

            Branch 2:
            [
              [1, 1, 0],  # Utterance 0 connects to itself and Utterance 1.
              [0, 1, 1],  # Utterance 1 connects to itself and Utterance 2.
              [0, 0, 1],  # Utterance 2 connects to itself only.
            ]
        """
        self.state_matrix = np.zeros((hps.branch_batch_size, hps.sen_batch_size, hps.sen_batch_size), dtype=np.int32)

        """
        Dialogue graphs often involve multiple speakers.
        For example, in a multi-party conversation, utterances from the same speaker may form a 
        coherent thread that the model needs to track.
        relate_user identifies and marks utterances spoken by the same speaker in chronological order.
        This is impt as speakers often exhibit certain patterns, preferences, or consistent styles across their utterances.

        For utterances  i and j in the same branch, if both are spoken by the same speaker and j occurs after 
        i, relate_user[branch][i][j] is set to 1.
        Otherwise, it is 0.
        [
          [0, 0, 1],  # Utterance 0 (Speaker A) is connected to Utterance 2 (Speaker A).
          [0, 0, 0],  # Utterance 1 (Speaker B) has no connections based on the speaker.
          [0, 0, 0],  # Utterance 2 (Speaker A) has no future utterances to connect to.
        ]
        """
        self.relate_user = np.zeros_like(self.state_matrix)

        """
        mask_emb is used to represent explicit relationships between the utterances in a dialogue branch.
        Each pair of related utterances in a branch is associated with a special embedding mask to highlight their connection.
        Each edge is represented with these masks.

        About the shape:
        sen_batch_size: For example, mask_emb[branch][i][j] corresponds to the relationship between utterance i (source) and j (target).
        sen_hidden_dim * 2 : This dimension encodes the relationship-specific features for the pair of utterances.

        For eg. Connected utterances are assigned with a pre-defined embedding (eg. mask_tool). Not connected utterances assigned 0 vector.
        Dialogue Graph for Branch 1:
          Utterance 0 → Utterance 1.
          Utterance 0 → Utterance 2.
        
        mask_emb for branch 1:
        [
          [[vector_0, vector_0, vector_0],  # Connections from utterance 0
          [mask_tool, vector_0, vector_0], # Connection from utterance 0 to 1
          [mask_tool, vector_0, vector_0]],# Connection from utterance 0 to 2

          [[vector_0, mask_tool, vector_0], # Connection from utterance 1 to 2
          [vector_0, vector_0, vector_0],  
          [vector_0, vector_0, vector_0]],
          
          ...
        ]
        """
        self.mask_emb = np.zeros((hps.branch_batch_size, hps.sen_batch_size, hps.sen_hidden_dim * 2), dtype=np.float32)

        # Initialize batch from tfexamples

        """
        i: Index of the current example in the batch (ranges from 0 to branch_batch_size - 1).
        ex: A single processed example, which contains:
          ex.enc_input: Encoder input for the dialogue branch (tokenized utterances).
          ex.dec_input: Decoder input (shifted ground truth sequence, starting with <start>).
          ex.dec_target: Target sequence (ground truth sequence, ending with <end>).
        """
        for i, ex in enumerate(tfexamples):
            # Fill encoder inputs
            for j, branch in enumerate(ex.enc_input):
                self.enc_batch[i, j, :] = branch[:]

            # Decoder inputs and targets
            self.dec_batch[i, :] = ex.dec_input
            self.target_batch[i, :] = ex.dec_target


class Batcher:
    """
    Handles batching of data and manages batch queues.
    """

    BATCH_QUEUE_MAX = 5

    """
    data_path: Path to the data files containing dialogue examples.
    vocab: Vocabulary object for tokenization and ID mapping.
    hps: Hyperparameters object, which includes details like branch_batch_size
    """
    def __init__(self, data_path, vocab, hps):
        self.data_path = data_path
        self.vocab = vocab
        self.hps = hps

        # Queues for input and batches
        self.batch_queue = Queue.Queue(self.BATCH_QUEUE_MAX)
        self.input_queue = Queue.Queue(self.BATCH_QUEUE_MAX * self.hps.branch_batch_size)

        # Start threads to manage data queues
        self.input_threads = [Thread(target=self._fill_input_queue) for _ in range(1)]
        self.batch_threads = [Thread(target=self._fill_batch_queue) for _ in range(1)]
        for thread in self.input_threads + self.batch_threads:
            thread.daemon = True
            thread.start()

    def _fill_input_queue(self):
        """
        Reads data from files, preprocesses it using record_maker, and adds the processed examples to input_queue.
        """
        while True:
            filelist = glob.glob(self.data_path)
            shuffle(filelist)
            for f in filelist:
                with open(f, 'rb') as reader:
                    for record in reader:
                        self.input_queue.put(record_maker(record, self.vocab, self.hps))

    def _fill_batch_queue(self):
        """
        Converts preprocessed examples from input_queue into batches and adds them to batch_queue.
        """
        while True:
            inputs = [self.input_queue.get() for _ in range(self.hps.branch_batch_size)]
            batches = [inputs[i:i + self.hps.branch_batch_size] for i in range(0, len(inputs), self.hps.branch_batch_size)]
            for b in batches:
                self.batch_queue.put(Batch(b, self.hps, self.vocab, None))


class record_maker:
    """
    Processes a single JSON record into structured input for the model.
    """

    def __init__(self, record, vocab, hps):
        # Parse JSON record
        record = json.loads(record)

        # Process context and response
        self.context = [vocab._word2id(w) for context in record['context'] for w in context.split()]
        self.response = [vocab._word2id(w) for w in record['answer'].split()]
        self.tgt_idx = record['ans_idx']  # Answer index

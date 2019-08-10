import tensorflow as tf
import numpy as np

import data

FLAGS = tf.app.flags.FLAGS

class Hypothesis(object):
  """Class to represent a hypothesis during beam search. Holds all the information needed for the hypothesis.
  """
  def __init__(self, tokens, log_probs, state, attn_dists):
    """Hypothesis constructor.
    """
    self.tokens = tokens
    self.log_probs = log_probs
    self.state = state
    self.attn_dists = attn_dists

  def extend(self, token, log_prob, state, attn_dist):
    """Return a NEW hypothesis, extended with the information from the latest step of beam search.
    """
    return Hypothesis(tokens = self.tokens + [token],
                      log_probs = self.log_probs + [log_prob],
                      state = state,
                      attn_dists = self.attn_dists + [attn_dist])

  @property
  def latest_token(self):
    return self.tokens[-1]

  @property
  def log_prob(self):
    return sum(self.log_probs)

  @property
  def avg_log_prob(self):
    return self.log_prob / len(self.tokens)


def run_beam_search(sess, model, vocab, batch):
  """Performs beam search decoding on the given example.
  """
  enc_states, dec_in_state = model._encode(sess, batch)
  hyps = [Hypothesis(tokens=[vocab._word2id(data.DECODING_START)],
                     log_probs=[0.0],
                     state=dec_in_state,
                     attn_dists=[]
                     ) for _ in xrange(FLAGS.beam_size)]
  results = []

  steps = 0
  while steps < FLAGS.max_dec_steps and len(results) < FLAGS.beam_size:
    latest_tokens = [h.latest_token for h in hyps]
    latest_tokens = [t if t in xrange(vocab._size()) else vocab._word2id(data.UNKNOWN_TOKEN) for t in latest_tokens]
    states = [h.state for h in hyps]

    (topk_ids, topk_log_probs, new_states, attn_dists) = model._decode(sess=sess,
                                                                       batch=batch,
                                                                       latest_tokens=latest_tokens,
                                                                       sen_enc_states=enc_states,
                                                                       dec_init_states=states)

    all_hyps = []
    num_orig_hyps = 1 if steps == 0 else len(hyps)
    for i in xrange(num_orig_hyps):
      h, new_state, attn_dist = hyps[i], new_states[i], attn_dists[i]
      for j in xrange(FLAGS.beam_size * 2):
        new_hyp = h.extend(token=topk_ids[i, j],
                           log_prob=topk_log_probs[i, j],
                           state=new_state,
                           attn_dist=attn_dist)
        all_hyps.append(new_hyp)

    hyps = []
    for h in sorted(all_hyps, key=lambda h: h.avg_log_prob, reverse=True):
      if h.latest_token == vocab._word2id(data.DECODING_END):
        if steps >= FLAGS.min_dec_steps:
          results.append(h)
      else:
        hyps.append(h)
      if len(hyps) == FLAGS.beam_size or len(results) == FLAGS.beam_size:
        break
    steps += 1

  if len(results)==0:
    results = hyps
  hyps_sorted = sorted(results, key = lambda h: h.avg_log_prob, reverse = True)

  return hyps_sorted[0]

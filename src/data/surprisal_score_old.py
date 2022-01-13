import torch
from lm_scorer.models.auto import AutoLMScorer as LMScorer

# Available models
list(LMScorer.supported_model_names())
# => ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl", distilgpt2"]

# Load model to cpu or cuda
device = "cuda:0" if torch.cuda.is_available() else "cpu"
batch_size = 1
model_path = r'D:\workspace_RES\MODEL\gpt2'
scorer = LMScorer.from_pretrained("gpt2", device=device, batch_size=batch_size)

# Return token probabilities (provide log=True to return log probabilities)
scorer.tokens_score("I like this package.")
# => (scores, ids, tokens)
# scores = [0.018321, 0.0066431, 0.080633, 0.00060745, 0.27772, 0.0036381]
# ids    = [40,       588,       428,      5301,       13,      50256]
# tokens = ["I",      "Ġlike",   "Ġthis",  "Ġpackage", ".",     "<|endoftext|>"]

# Compute sentence score as the product of tokens' probabilities
scorer.sentence_score("I like this package.", reduce="prod")
# => 6.0231e-12

# Compute sentence score as the mean of tokens' probabilities
scorer.sentence_score("I like this package.", reduce="mean")
# => 0.064593

# Compute sentence score as the geometric mean of tokens' probabilities
scorer.sentence_score("I like this package.", reduce="gmean")
# => 0.013489

# Compute sentence score as the harmonic mean of tokens' probabilities
scorer.sentence_score("I like this package.", reduce="hmean")
# => 0.0028008

# Get the log of the sentence score.
scorer.sentence_score("I like this package.", log=True)
# => -25.835

# Score multiple sentences.
scorer.sentence_score(["Sentence 1", "Sentence 2"])
# => [1.1508e-11, 5.6645e-12]

# NB: Computations are done in log space so they should be numerically stable.
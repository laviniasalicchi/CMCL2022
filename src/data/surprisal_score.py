import math
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from transformers import GPT2LMHeadModel, GPT2TokenizerFast
device = 'cuda'
model_id = 'gpt2-large'
model_path = r'D:\workspace_RES\MODEL\gpt2'
model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
# model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
tokenizer = GPT2TokenizerFast.from_pretrained(model_path)

from nlp import load_dataset
test = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
encodings = tokenizer('\n\n'.join(test['text']), return_tensors='pt')

# a=['there is a book on the desk',
#    'there is a plane on the desk',
#    'there is a book in the desk']
# encodings = tokenizer(a, return_tensors='pt')

max_length = model.config.n_positions
stride = 512

lls = []
for i in range(0, encodings.input_ids.size(1), stride):
    begin_loc = max(i + stride - max_length, 0)
    end_loc = min(i + stride, encodings.input_ids.size(1))
    trg_len = end_loc - i    # may be different from stride on last loop
    input_ids = encodings.input_ids[:,begin_loc:end_loc].to(device)
    target_ids = input_ids.clone()
    target_ids[:,:-trg_len] = -100

    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)
        log_likelihood = outputs[0] * trg_len

    lls.append(log_likelihood)

ppl = torch.exp(torch.stack(lls).sum() / end_loc)
print(ppl)

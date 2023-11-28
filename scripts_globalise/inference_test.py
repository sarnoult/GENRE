import pickle
from genre.trie import Trie, MarisaTrie

with open("../data/training/lang_title_globalise.pkl", "rb") as f:
    trie = pickle.load(f)

from genre.fairseq_model import mGENRE
model = mGENRE.from_pretrained("models/fairseq_multilingual_entity_disambiguation").eval()

# for huggingface/transformers
# from genre.hf_model import mGENRE
# model = mGENRE.from_pretrained("../models/hf_multilingual_entity_disambiguation").eval()


sentences = ["ontfangst der reek: van des irmanse [START] wvol [END] ."]
#,
#             "wat quantiteijt er nog op weg is, waer na de [START] scheepen [END] niet derven"]

model.sample(
    sentences,
    prefix_allowed_tokens_fn=lambda batch_id, sent: [
        e for e in trie.get(sent.tolist())
        if e < len(model.task.target_dictionary)
        # for huggingface/transformers
        # if e < len(model2.tokenizer) - 1
    ],
)
import pickle
import sys
from genre.trie import Trie, MarisaTrie

from genre.fairseq_model import mGENRE

def main(pkl_trie, test_sentences):
    with open(pkl_trie, "rb") as f:
        trie = Trie.load_from_dict(pickle.load(f))

    model = mGENRE.from_pretrained("models/fairseq_multilingual_entity_disambiguation").eval()

    with open(test_sentences) as f:
        sentences = list(map(lambda x: x.strip(), f.readlines()))

    output = model.sample(
        sentences,
        prefix_allowed_tokens_fn=lambda batch_id, sent: [
            e for e in trie.get(sent.tolist())
            if e < len(model.task.target_dictionary)
            # for huggingface/transformers
            # if e < len(model2.tokenizer) - 1
        ],
    )
    print(output)

if __name__ == '__main__':
    targets_pkl = sys.argv[1]   # "../data/training/lang_title_globalise.pkl"
    test_file = sys.argv[2]
    main(targets_pkl, test_file)
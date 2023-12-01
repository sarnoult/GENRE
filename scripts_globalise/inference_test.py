import pickle
import sys
from genre.trie import Trie, MarisaTrie

from genre.fairseq_model import mGENRE

def main(pkl_trie, test_sentences, targets_out):
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
        ],
    )
    with open(targets_out_pkl, "rb") as f:
        pickle.dump(output, f)

if __name__ == '__main__':
    targets_pkl = sys.argv[1]   
    test_file = sys.argv[2]
    targets_out_pkl = sys.argv[3]
    main(targets_pkl, test_file, targets_out_pkl)
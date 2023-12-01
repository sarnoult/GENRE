import argparse
import pickle
import sys
from genre.trie import Trie, MarisaTrie

from genre.fairseq_model import mGENRE

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "targets_trie",
        type=str,
        help="targets trie for constrained search"
    )
    parser.add_argument(
        "source",
        help="source file (text file)",
        type=str,
    )
    parser.add_argument(
        "targets",
        help="target predictions (pkl file)",
        type=str,
    )
    parser.add_argument(
        "-n",
        "--beam_size",
        type=int,
        help="beam search size / number of predictions",
        default=10
    )
    parser.add_argument(
        "-m",
        "--model_dir",
        help="Pretrained model directory",
        type=str,
        default="models/fairseq_multilingual_entity_disambiguation"
    )
    args = parser.parse_args()
    
    with open(args.targets_trie, "rb") as f:
        trie = Trie.load_from_dict(pickle.load(f))

    model = mGENRE.from_pretrained(args.model_dir).eval()

    with open(args.source) as f:
        sentences = list(map(lambda x: x.strip(), f.readlines()))

    output = model.sample(
        sentences,
        prefix_allowed_tokens_fn=lambda batch_id, sent: [
            e for e in trie.get(sent.tolist())
            if e < len(model.task.target_dictionary)
        ],
        beam=args.beam_size
    )
    with open(args.targets, "wb") as f:
        pickle.dump(output, f)

if __name__ == '__main__':
    main()
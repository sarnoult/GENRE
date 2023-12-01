import argparse
from collections import defaultdict
import logging
import os
import pickle


from genre.fairseq_model import mGENRE
from genre.utils import chunk_it, extract_pages
from genre.trie import Trie
from tqdm.auto import tqdm, trange


def add_to_trie(seq, dct):
    Trie._add_to_trie(seq, trie_dict=dct)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "kb_titles",
        type=str,
        help="file of KB titles"
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        help="Output directory for titles trie",
        type=str,
    )
    parser.add_argument(
        "-m",
        "--model_dir",
        help="Pretrained model directory",
        type=str,
        default="models/fairseq_multilingual_entity_disambiguation"
    )
    parser.add_argument(
        "-d",
        "--debug",
        help="Print lots of debugging statements",
        action="store_const",
        dest="loglevel",
        const=logging.DEBUG,
        default=logging.WARNING,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="Be verbose",
        action="store_const",
        dest="loglevel",
        const=logging.INFO,
    )

    args = parser.parse_args()

    logging.basicConfig(level=args.loglevel)


    mgenre = mGENRE.from_pretrained(
        args.model_dir,
        checkpoint_file="model.pt",
        bpe="sentencepiece",
        layernorm_embedding=True,
        sentencepiece_model="spm_256000.model",
    ).eval()

    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.kb_titles)

    with open(args.kb_titles) as f:
        titles = f.readlines()
    titles2bpes = {
        title.strip(): mgenre.encode(title.strip()).tolist() 
            for title in titles
    }
    titles_bpes = defaultdict(dict)
    for title, bpes in tqdm(titles2bpes.items()):
        titles_bpes[title] = [2] + bpes[1:] 
    #print(mgenre.encode(titles2bpes[titles[0].strip()]).tolist())
    # with open(os.path.join(args.output_dir, "lang_title_globalise.pkl"), "wb") as f:
    #     pickle.dump(lang_titles2bpes, f)

    trie = {}
    for sequence in tqdm(titles_bpes.values()):
        add_to_trie(sequence, trie)

    logging.info("Saving {}".format(os.path.join(args.output_dir, "globalise_titles_trie.pkl")))
    with open(os.path.join(args.output_dir, "globalise_titles_trie.pkl"), "wb") as f:
        pickle.dump(trie, f)

    

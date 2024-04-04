from datasets import load_dataset
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset-cache-dir",
    type=str,
    required=True,
    help="Path to save the pile dataset",
)

args = parser.parse_args()

raw_datasets = load_dataset('monology/pile-uncopyrighted')

raw_datasets.save_to_disk(args.dataset_cache_dir, num_proc=64)


# filtered data for evaluation

# pg19
pg19 = raw_datasets.filter(lambda item: item['meta']['pile_set_name'] == 'Gutenberg (PG-19)', num_proc=64)
pg19.save_to_disk(args.dataset_cache_dir+"_pg19", num_proc=64)

# github
github = raw_datasets.filter(lambda item: item['meta']['pile_set_name'] == 'Github', num_proc=64)
github.save_to_disk(args.dataset_cache_dir+"_pg19", num_proc=64)


# aixiv
arxiv = raw_datasets.filter(lambda item: item['meta']['pile_set_name'] == 'ArXiv', num_proc=64)
arxiv.save_to_disk(args.dataset_cache_dir+"_arxiv", num_proc=64)
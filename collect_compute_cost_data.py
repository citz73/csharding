import argparse
import os

from neuroshard.compute_bench import benchmark_compute
from neuroshard.utils import load_dlrm_dataset

def main():
    parser = argparse.ArgumentParser("NeuroShard compute cost data collection")
    parser.add_argument("--data_size", type=int, default=999999999)
    parser.add_argument("--num_cpus", type=int, default=4) # is this CPU?
    parser.add_argument("--max_tables", type=int, default=15)
    parser.add_argument("--max_mem", type=int, default=4)
    parser.add_argument("--max_dim", type=int, default=128)
    parser.add_argument("--dataset_dir", type=str, default="data/dlrm_datasets")
    parser.add_argument("--out_dir", type=str, default="data/cost_data/compute")
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    table_configs, data = load_dlrm_dataset(args.dataset_dir)

    benchmark_compute(
        data_size=args.data_size,
        num_cpus=args.num_cpus,
        max_tables=args.max_tables,
        max_mem=args.max_mem,
        max_dim=args.max_dim,
        table_configs=table_configs,
        data=data,
        out_dir=args.out_dir,
    )

if __name__ == '__main__':
    main()


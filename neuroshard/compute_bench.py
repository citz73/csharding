import copy
import json
import gc
import os
import traceback
from typing import Dict, Any, Callable

import numpy as np
import torch
from torch import multiprocessing as mp
from fbgemm_gpu import split_table_batched_embeddings_ops # TODO
from fbgemm_gpu.split_embedding_configs import EmbOptimType as OptimType, SparseType # TODO

from neuroshard.utils import (
    Timer,
    table_size,
    get_data,
)

def benchmark_compute(
    data_size,
    num_cpus,
    max_tables,
    max_mem,
    max_dim,
    table_configs,
    data,
    out_dir,
):
    torch.set_num_threads(1)

    # Augment the tables with different table dims
    aug_table_configs = []
    cur_dim = max_dim
    while cur_dim % 4 == 0:
        cur_table_configs = copy.deepcopy(table_configs)
        for config in cur_table_configs:
            config["dim"] = cur_dim
        aug_table_configs.extend(cur_table_configs)
        cur_dim //= 2

    # Save the augmented table_configs
    with open(os.path.join(out_dir, "table_configs.json"), "w") as f:
        json.dump(aug_table_configs, f)

    def gen_task():
        size = np.random.randint(low=1, high=max_tables+1)
        table_indices = np.random.randint(len(aug_table_configs), size=size).tolist()
        while not sum([table_size(aug_table_configs[i]["row"], aug_table_configs[i]["dim"]) for i in table_indices]) <= max_mem:
            table_indices = np.random.randint(len(aug_table_configs), size=size).tolist()
        return table_indices


    # Create a Pool with the desired number of CPU cores
    with mp.Pool(processes=num_cpus) as pool:
        # Initialize task_queues and result_queue
        manager = mp.Manager()
        task_queues = [manager.Queue() for _ in range(num_cpus)]
        result_queue = manager.Queue()

        # Start worker processes
        processes = [
            pool.apply_async(benchmark_compute_internal, args=(
                i,
                aug_table_configs,
                data,
                task_queues[i],
                result_queue,
            ), error_callback=lambda e: print(e))
            for i in range(num_cpus)
        ]

        # Initial assignment
        task_id = 0
        while task_id < num_cpus:
            task_queues[task_id].put(gen_task())
            task_id += 1
            print(f'task being queued {task_id}')
        num_running_devices = num_cpus

        # Assign once finished
        with open(os.path.join(out_dir, "data.txt"), "a") as f:
            while True:
                task, cost, i = result_queue.get()
                task = ",".join(list(map(str, task)))
                data = f"task: {task} | cost: {cost}"
                print(data)
                f.write(data + "\n")
                f.flush()
                if task_id < data_size:
                    task_queues[i].put(gen_task())
                    task_id += 1
                else:
                    task_queues[i].put(-1)
                    num_running_devices -= 1
                if num_running_devices == 0:
                    break

def benchmark_compute_internal(
    process_index,
    table_configs,
    data,
    task_queue,
    result_queue,
):
    cpu_id = mp.current_process().name
    print(f"Process {process_index} is running on CPU(s): {cpu_id}")
    # Benchmarking
    compute_bench = ComputeBench(
        table_configs=table_configs,
        data=data,
    )
    torch.set_num_threads(1)
    print(f"Process {cpu_id} initialized!")

    try:
        while True:
            task = task_queue.get()
            if task == -1:
                break
            cost = compute_bench.eval(task)
            result_queue.put((task, cost, process_index))
        print(f"Process {cpu_id} finished!")
    except KeyboardInterrupt:
        pass  # Return silently.
    except Exception as e:
        print(f"Exception in process {cpu_id}")
        traceback.print_exc()
        print()
        raise e

class ComputeBench:
    def __init__(
        self,
        table_configs,
        data,
        warmup_iter=5,
        num_iter=10,
        device="cpu", #changed
    ):

        self.table_configs = table_configs

        # Get indices and offsets
        self.offsets = data["offsets"]
        self.indices = data["indices"]
        self.num_data = len(self.offsets)

        self.warmup_iter = warmup_iter
        self.num_iter = num_iter
        self.device = device
        self.batch_size = self.offsets[0].shape[0] - 1

    def eval(self, table_indices):
        print('at eval')
        if len(table_indices) == 0:
            return 0

        gc.collect()

        # Build the op
        shard_table_configs = [self.table_configs[i] for i in table_indices]

        op = split_table_batched_embeddings_ops.SplitTableBatchedEmbeddingBagsCodegen( # TODO
            [
                (
                    table_config["row"],
                    table_config["dim"],
                    split_table_batched_embeddings_ops.EmbeddingLocation.HOST,
                    split_table_batched_embeddings_ops.ComputeDevice.CPU,
                )
                for table_config in shard_table_configs
            ],
            optimizer=OptimType.EXACT_SGD,
            cache_algorithm=split_table_batched_embeddings_ops.CacheAlgorithm.LFU,
            cache_reserved_memory=8.0,
            eps=0.01,
            device=self.device,
            weights_precision=SparseType.FP32,
        )
        print('at eval:data')

        # Get data
        shard_offsets = [self.offsets[i%self.num_data] for i in table_indices]
        shard_indices = [self.indices[i%self.num_data] for i in table_indices]
        print('HERE')
        args, kwargs, grads_tensor = get_data(
            self.batch_size,
            shard_offsets,
            shard_indices,
            sum([self.table_configs[index]["dim"] for index in table_indices]),
            self.device,
        )
        print('at eval: after shard')
        time_records = benchmark_op(
            op,
            args,
            kwargs,
            grads_tensor,
            self.device,
            num_iter=self.warmup_iter+self.num_iter,
        )[self.warmup_iter:]
        print('at eval: after time')
        return np.median(time_records, axis=0)

def benchmark_op(op: Callable, args: Any, kwargs: Any, grads_tensor: Any, device: str, num_iter: int):
    time_records = []
    print(grads_tensor.shape)
    for _ in range(num_iter):
        _ = torch.rand(6 * 1024 * 1024 // 4).float() * 2  # V100 6MB L2 cache

        with Timer(device) as timer:
            op(*args, **kwargs).backward(grads_tensor) # FAILS
        time_records.append(timer.elapsed_time() * 1000)
    return time_records


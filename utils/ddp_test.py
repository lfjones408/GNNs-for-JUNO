import torch
import torch.distributed as dist
import os
import datetime

def main():
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=3600))
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    
    # Simple all-reduce test
    tensor = torch.ones(1).cuda() * rank
    dist.all_reduce(tensor)
    print(f"Rank {rank}: {tensor.item()}")

if __name__ == "__main__":
    main()
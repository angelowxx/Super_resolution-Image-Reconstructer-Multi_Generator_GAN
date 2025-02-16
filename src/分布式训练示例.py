import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist

def train(rank, world_size):
    # 初始化进程组
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # 设置 GPU
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # 初始化模型
    model = YourModel().to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # 数据加载
    dataset = YourDataset()
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, sampler=sampler)

    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)  # 保证不同 GPU 训练的数据不重复
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    dist.destroy_process_group()  # 训练结束后销毁进程组

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size,), nprocs=world_size)
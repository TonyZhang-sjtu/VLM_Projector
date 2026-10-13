import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from src.training.loss import global_answer_tokens,backward_scale


def worker(rank,init,output):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+init,rank=rank,world_size=2)
    torch.manual_seed(17)
    model=torch.nn.Linear(4,7);wrapped=DDP(model)
    x=torch.arange(48,dtype=torch.float32).reshape(12,4)/48
    y=torch.arange(12)%7
    indices=slice(0,3) if rank==0 else slice(3,12)
    local=x[indices];target=y[indices]
    total=global_answer_tokens(len(target),'cpu')
    loss=torch.nn.functional.cross_entropy(wrapped(local),target,reduction='sum')*backward_scale(total,2)
    loss.backward()
    if rank==0:
        torch.save([p.grad for p in model.parameters()],output)
    dist.destroy_process_group()


def test_actual_ddp_answer_token_scaling(tmp_path):
    mp.spawn(worker,args=(str(tmp_path/'init'),str(tmp_path/'grads.pt')),nprocs=2,join=True)
    torch.manual_seed(17);model=torch.nn.Linear(4,7)
    x=torch.arange(48,dtype=torch.float32).reshape(12,4)/48;y=torch.arange(12)%7
    torch.nn.functional.cross_entropy(model(x),y).backward()
    grads=torch.load(tmp_path/'grads.pt',weights_only=True)
    for actual,p in zip(grads,model.parameters()):torch.testing.assert_close(actual,p.grad,rtol=1e-5,atol=1e-7)

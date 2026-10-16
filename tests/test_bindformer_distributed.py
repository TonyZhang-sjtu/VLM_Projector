"""Real two-process DDP: CE is per answer token, reconstruction per image."""
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from src.training.loss import combined_loss,global_answer_tokens,global_sample_count


def worker(rank,init,output):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+init,rank=rank,world_size=2)
    torch.manual_seed(18)
    model=torch.nn.Linear(4,7);wrapped=DDP(model)
    x=torch.arange(48,dtype=torch.float32).reshape(12,4)/48
    y=torch.arange(12)%7
    sl=slice(0,3) if rank==0 else slice(3,12)
    logits=wrapped(x[sl])
    # Rank 0 has one image/3 tokens, rank 1 has two images/9 tokens.
    feat=logits[0].square().mean()
    if rank==1:feat=feat+logits[4].square().mean()
    result={'loss_sum':torch.nn.functional.cross_entropy(logits,y[sl],reduction='sum'),
            'feature_loss_sum':feat}
    count=global_answer_tokens(len(y[sl]),'cpu')
    samples=global_sample_count(rank+1,'cpu')
    combined_loss(result,count,2,.2,samples).backward()
    if rank==0:torch.save([p.grad for p in model.parameters()],output)
    dist.destroy_process_group()


def test_ddp_unequal_tokens_and_images_match_single_batch(tmp_path):
    mp.spawn(worker,args=(str(tmp_path/'init'),str(tmp_path/'grads.pt')),nprocs=2,join=True)
    torch.manual_seed(18);model=torch.nn.Linear(4,7)
    x=torch.arange(48,dtype=torch.float32).reshape(12,4)/48;y=torch.arange(12)%7
    logits=model(x)
    loss=torch.nn.functional.cross_entropy(logits,y)+.2*logits[[0,3,7]].square().mean()
    loss.backward()
    grads=torch.load(tmp_path/'grads.pt',weights_only=True)
    for actual,p in zip(grads,model.parameters()):
        torch.testing.assert_close(actual,p.grad,rtol=1e-5,atol=1e-7)

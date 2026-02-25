import torch, os, time, random, generator, discri, classify, utils
import csv
import numpy as np 
import torch.nn as nn
import torchvision.utils as tvls
import torch.nn.functional as F
from utils import log_sum_exp, save_tensor_images
from torch.autograd import Variable
import torch.optim as optim
import torch.autograd as autograd
import statistics 
import torch.distributions as tdist

device = "cuda"
num_classes = 1000
save_img_dir = './res_all' # all attack imgs
os.makedirs(save_img_dir, exist_ok=True)
success_dir = './res_success'
os.makedirs(success_dir, exist_ok=True)


def reparameterize(mu, logvar):
    """
    Reparameterization trick to sample from N(mu, var) from
    N(0,1).
    :param mu: (Tensor) Mean of the latent Gaussian [B x D]
    :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
    :return: (Tensor) [B x D]
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)

    return eps * std + mu


def dist_inversion(G, D, T, E, iden, itr, lr=2e-2, momentum=0.9, lamda=100, iter_times=1500, clip_range=1, improved=False, num_seeds=5, run_type="base"):
    iden = iden.view(-1).long().cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    bs = iden.shape[0]
    
    G.eval()
    D.eval()
    T.eval()
    E.eval()

    no = torch.zeros(bs) # index for saving all success attack images

    # CSV logging (one file per run_type)
    acc_dir = './acc_results'
    os.makedirs(acc_dir, exist_ok=True)

    def _safe_name(s):
        s = str(s)
        return ''.join(c if (c.isalnum() or c in ['-', '_']) else '_' for c in s)

    run_id = "{}_itr{}_imp{}".format(time.strftime('%Y%m%d-%H%M%S'), itr, int(improved))
    csv_path = os.path.join(acc_dir, "dist_inversion_{}.csv".format(_safe_name(run_type)))
    csv_header = [
        'run_id', 'timestamp', 'itr', 'run_type', 'improved', 'bs', 'iter_times', 'lr', 'momentum', 'lamda', 'clip_range',
        'event', 'iteration', 'seed', 'prior_loss', 'iden_loss', 'attack_acc',
        'acc_top1', 'acc_top5',
        'acc_mean_top1', 'acc_mean_top5', 'acc_var_top1', 'acc_var_top5',
        'elapsed_sec'
    ]

    def _append_row(**kwargs):
        file_exists = os.path.exists(csv_path)
        needs_header = (not file_exists) or (os.path.getsize(csv_path) == 0)
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if needs_header:
                writer.writerow(csv_header)

            base = {
                'run_id': run_id,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'itr': int(itr),
                'run_type': str(run_type),
                'improved': int(improved),
                'bs': int(bs),
                'iter_times': int(iter_times),
                'lr': float(lr),
                'momentum': float(momentum),
                'lamda': float(lamda),
                'clip_range': float(clip_range),
                'event': '',
                'iteration': '',
                'seed': '',
                'prior_loss': '',
                'iden_loss': '',
                'attack_acc': '',
                'acc_top1': '',
                'acc_top5': '',
                'acc_mean_top1': '',
                'acc_mean_top5': '',
                'acc_var_top1': '',
                'acc_var_top5': '',
                'elapsed_sec': '',
            }
            base.update(kwargs)
            writer.writerow([base.get(k, '') for k in csv_header])

    opt_start_time = time.time()

    #NOTE
    mu = Variable(torch.zeros(bs, 100), requires_grad=True)
    log_var = Variable(torch.ones(bs, 100), requires_grad=True)
    
    params = [mu, log_var]
    solver = optim.Adam(params, lr=lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(solver, 1800, gamma=0.1)
        
    for i in range(iter_times):
        z = reparameterize(mu, log_var)
        fake = G(z)
        if improved == True:
            _, label =  D(fake)
        else:
            label = D(fake)
        
        out = T(fake)[-1]

        # "Black-box" variants: build tensors without in-place ops on autograd outputs.
        out_soft = torch.softmax(out, dim=1)
        if run_type == 'bb':
            output = out_soft
        elif run_type == 'bb_top1':
            # Top-1 hard label, but keep gradients flowing like `out_soft` (straight-through estimator).
            max_idx = out_soft.argmax(dim=1, keepdim=True)
            one_hot = torch.zeros_like(out_soft).scatter(1, max_idx, 1.0)
            output = one_hot
        elif run_type == 'bb_top5':
            _, topk_idx = torch.topk(out_soft, k=5, dim=1, largest=True, sorted=True)
            fixed = torch.tensor([0.24, 0.22, 0.20, 0.18, 0.16], device=out_soft.device, dtype=out_soft.dtype)
            fixed = fixed.view(1, 5).expand(bs, 5)
            output = torch.zeros_like(out_soft).scatter(1, topk_idx, fixed)
        else:
            # "White-box" varinat
            output = out
        
        Iden_Loss = criterion(output, iden)

        for p in params:
            if p.grad is not None:
                p.grad.data.zero_()

        if improved:
            Prior_Loss = torch.mean(F.softplus(log_sum_exp(label))) - torch.mean(log_sum_exp(label))
            # Prior_Loss =  torch.mean(F.softplus(log_sum_exp(label))) - torch.mean(label.gather(1, iden.view(-1, 1)))  #1 class prior
        else:
            Prior_Loss = - label.mean()

        Total_Loss = Prior_Loss + lamda * Iden_Loss

        Total_Loss.backward()
        solver.step()
        
        z = torch.clamp(z.detach(), -clip_range, clip_range).float()

        Prior_Loss_val = Prior_Loss.item()
        Iden_Loss_val = Iden_Loss.item()

        if (i+1) % 300 == 0:
            fake_img = G(z.detach())
            eval_prob = E(utils.low2high(fake_img))[-1]
            eval_iden = torch.argmax(eval_prob, dim=1).view(-1)
            acc = iden.eq(eval_iden.long()).sum().item() * 1.0 / bs
            print("Iteration:{}\tPrior Loss:{:.2f}\tIden Loss:{:.2f}\tAttack Acc:{:.2f}".format(i+1, Prior_Loss_val, Iden_Loss_val, acc))
                _append_row(
                event='progress',
                iteration=int(i + 1),
                prior_loss=float(Prior_Loss_val),
                iden_loss=float(Iden_Loss_val),
                attack_acc=float(acc),
                elapsed_sec=float(time.time() - opt_start_time),
                )
            
            opt_interval = time.time() - opt_start_time
            print("Time:{:.2f}".format(opt_interval))
            _append_row(event='time', elapsed_sec=float(opt_interval))
    
    res = []
    res5 = []
    seed_acc = torch.zeros((bs, num_seeds))
    for random_seed in range(num_seeds):
        seed_start_time = time.time()
        z = reparameterize(mu, log_var)
        fake = G(z)
        score = T(fake)[-1]
        eval_prob = E(utils.low2high(fake))[-1]
        eval_iden = torch.argmax(eval_prob, dim=1).view(-1)
        
        cnt, cnt5 = 0, 0
        for i in range(bs):
            gt = iden[i].item()
            sample = fake[i]
            save_tensor_images(sample.detach(), os.path.join(save_img_dir, "attack_iden_{}_{}.png".format(gt+1, random_seed)))

            if eval_iden[i].item() == gt:
                seed_acc[i, random_seed] = 1
                cnt += 1
                best_img = G(z)[i]
                save_tensor_images(best_img.detach(), os.path.join(success_dir, "{}_attack_iden_{}_{}.png".format(itr, gt+1, int(no[i]))))
                no[i] += 1
            _, top5_idx = torch.topk(eval_prob[i], 5)
            if gt in top5_idx:
                cnt5 += 1
                
        seed_interval = time.time() - seed_start_time
        top1_acc = cnt * 1.0 / bs
        top5_acc = cnt5 * 1.0 / bs
        print("Time:{:.2f}\tSeed:{}\tAcc:{:.2f}\t".format(seed_interval, random_seed, top1_acc))
        _append_row(
            event='seed',
            seed=int(random_seed),
            acc_top1=float(top1_acc),
            acc_top5=float(top5_acc),
            elapsed_sec=float(seed_interval),
        )
        res.append(cnt * 1.0 / bs)
        res5.append(cnt5 * 1.0 / bs)

        torch.cuda.empty_cache()
        
    
    acc, acc_5 = statistics.mean(res), statistics.mean(res5)
    acc_var = statistics.variance(res)
    acc_var5 = statistics.variance(res5)
    print("Acc:{:.2f}\tAcc_5:{:.2f}\tAcc_var:{:.4f}\tAcc_var5:{:.4f}".format(acc, acc_5, acc_var, acc_var5))

    _append_row(
        event='summary',
        acc_mean_top1=float(acc),
        acc_mean_top5=float(acc_5),
        acc_var_top1=float(acc_var),
        acc_var_top5=float(acc_var5),
        elapsed_sec=float(time.time() - opt_start_time),
    )


    return acc, acc_5, acc_var, acc_var5


def inversion(G, D, T, E, iden, itr, lr=2e-2, momentum=0.9, lamda=100, iter_times=1500, clip_range=1, improved=False, num_seeds=5):
	iden = iden.view(-1).long().cuda()
	criterion = nn.CrossEntropyLoss().cuda()
	bs = iden.shape[0]
	
	G.eval()
	D.eval()
	T.eval()
	E.eval()

	flag = torch.zeros(bs)
	no = torch.zeros(bs) # index for saving all success attack images

	res = []
	res5 = []
	seed_acc = torch.zeros((bs, 5))
	for random_seed in range(num_seeds):
		tf = time.time()
		r_idx = random_seed
		torch.manual_seed(random_seed) 
		torch.cuda.manual_seed(random_seed) 
		np.random.seed(random_seed) 
		random.seed(random_seed)

		z = torch.randn(bs, 100).cuda().float()
		z.requires_grad = True
		v = torch.zeros(bs, 100).cuda().float()
			
		for i in range(iter_times):
			fake = G(z)
			if improved == True:
				_, label =  D(fake)
			else:
				label = D(fake)
			
			out = T(fake)[-1]
			
			
			if z.grad is not None:
				z.grad.data.zero_()

			if improved:
				Prior_Loss = torch.mean(F.softplus(log_sum_exp(label))) - torch.mean(log_sum_exp(label))
				# Prior_Loss =  torch.mean(F.softplus(log_sum_exp(label))) - torch.mean(label.gather(1, iden.view(-1, 1)))  #1 class prior
			else:
				Prior_Loss = - label.mean()

			Iden_Loss = criterion(out, iden)

			Total_Loss = Prior_Loss + lamda * Iden_Loss

			Total_Loss.backward()
			
			v_prev = v.clone()
			gradient = z.grad.data
			v = momentum * v - lr * gradient
			z = z + ( - momentum * v_prev + (1 + momentum) * v)
			z = torch.clamp(z.detach(), -clip_range, clip_range).float()
			z.requires_grad = True

			Prior_Loss_val = Prior_Loss.item()
			Iden_Loss_val = Iden_Loss.item()

			if (i+1) % 300 == 0:
				fake_img = G(z.detach())
				eval_prob = E(utils.low2high(fake_img))[-1]
				eval_iden = torch.argmax(eval_prob, dim=1).view(-1)
				acc = iden.eq(eval_iden.long()).sum().item() * 1.0 / bs
				print("Iteration:{}\tPrior Loss:{:.2f}\tIden Loss:{:.2f}\tAttack Acc:{:.2f}".format(i+1, Prior_Loss_val, Iden_Loss_val, acc))
			
		fake = G(z)
		score = T(fake)[-1]
		eval_prob = E(utils.low2high(fake))[-1]
		eval_iden = torch.argmax(eval_prob, dim=1).view(-1)
		
		cnt, cnt5 = 0, 0
		for i in range(bs):
			gt = iden[i].item()
			sample = G(z)[i]
			# save_tensor_images(sample.detach(), os.path.join(save_img_dir, "attack_iden_{}_{}.png".format(gt+1, r_idx)))

			if eval_iden[i].item() == gt:
				seed_acc[i, r_idx] = 1
				cnt += 1
				flag[i] = 1
				best_img = G(z)[i]
				# save_tensor_images(best_img.detach(), os.path.join(success_dir, "{}_attack_iden_{}_{}.png".format(itr, iden[0]+i+1, int(no[i]))))
				no[i] += 1
			_, top5_idx = torch.topk(eval_prob[i], 5)
			if gt in top5_idx:
				cnt5 += 1
				
		
		interval = time.time() - tf
		print("Time:{:.2f}\tAcc:{:.2f}\t".format(interval, cnt * 1.0 / bs))
		res.append(cnt * 1.0 / bs)
		res5.append(cnt5 * 1.0 / bs)
		torch.cuda.empty_cache()

	acc, acc_5 = statistics.mean(res), statistics.mean(res5)
	acc_var = statistics.variance(res)
	acc_var5 = statistics.variance(res5)
	print("Acc:{:.2f}\tAcc_5:{:.2f}\tAcc_var:{:.4f}\tAcc_var5:{:.4f}".format(acc, acc_5, acc_var, acc_var5))
	
	return acc, acc_5, acc_var, acc_var5
import os
import time
import utils
import torch
import dataloader
import torchvision
from utils import *
from torch.nn import BCELoss
from torch.autograd import grad
import torchvision.utils as tvls
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.functional as F
from discri import DGWGAN, Discriminator, MinibatchDiscriminator
from generator import Generator
from classify import *
try:
    from tensorboardX import SummaryWriter  # type: ignore
except ImportError:
    from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())

def freeze(net):
    for p in net.parameters():
        p.requires_grad_(False) 

def unfreeze(net):
    for p in net.parameters():
        p.requires_grad_(True)

def gradient_penalty(x, y):
    # interpolation
    shape = [x.size(0)] + [1] * (x.dim() - 1)
    alpha = torch.rand(shape).cuda()
    z = x + alpha * (y - x)
    z = z.cuda()
    z.requires_grad = True

    o = DG(z)
    g = grad(o, z, grad_outputs = torch.ones(o.size()).cuda(), create_graph = True)[0].view(z.size(0), -1)
    gp = ((g.norm(p = 2, dim = 1) - 1) ** 2).mean()

    return gp

def log_sum_exp(x, axis = 1):
    m = torch.max(x, dim = 1)[0]
    return m + torch.log(torch.sum(torch.exp(x - m.unsqueeze(1)), dim = axis))


save_img_dir = "./improvedGAN/imgs_improved_celeba_gan"
save_model_dir= "./improvedGAN/"
os.makedirs(save_model_dir, exist_ok=True)
os.makedirs(save_img_dir, exist_ok=True)

dataset_name = "celeba"

log_path = "./improvedGAN/attack_logs"
os.makedirs(log_path, exist_ok=True)
log_file = "improvedGAN_celeba.txt"
utils.Tee(os.path.join(log_path, log_file), 'w')



if __name__ == "__main__":
    global args, writer
    
    file = "./config/" + dataset_name + ".json"
    args = load_json(json_file=file)
    writer = SummaryWriter(log_path)

    file_path = args['dataset']['gan_file_path']
    model_name = args['dataset']['model_name']
    lr = args[model_name]['lr']
    batch_size = args[model_name]['batch_size']
    z_dim = args[model_name]['z_dim']
    epochs = args[model_name]['epochs']
    n_critic = args[model_name]['n_critic']

    # Pick the best VGG16 checkpoint by parsing accuracy from filename.
    # Expected pattern: VGG16_<ACC>_*.tar or VGG16_<ACC>.tar
    ckpt_dir = './target_model/target_ckp'
    best_acc = None
    path_T = None
    for fname in os.listdir(ckpt_dir):
        if not (fname.startswith('VGG16_') and fname.endswith('.tar')):
            continue
        parts = fname.split('_')
        if len(parts) < 2:
            continue
        # parts[1] is the token between 1st and 2nd underscore.
        # It may be like "88.26.tar" or "74.48" depending on filename.
        acc_token = parts[1]
        if acc_token.endswith('.tar'):
            acc_token = acc_token[:-4]
        try:
            acc = float(acc_token)
        except ValueError:
            continue
        if best_acc is None or acc > best_acc:
            best_acc = acc
            path_T = os.path.join(ckpt_dir, fname)

    if path_T is None:
        raise FileNotFoundError(f'No suitable VGG16_*.tar found in {ckpt_dir}')

    print(f"Selected target checkpoint: {path_T} (acc={best_acc:.2f})")
    T = VGG16(1000)


    T = torch.nn.DataParallel(T).cuda()
    ckp_T = torch.load(path_T)
    T.load_state_dict(ckp_T['state_dict'], strict=False)

    print("---------------------Training [%s]------------------------------" % model_name)
    utils.print_params(args["dataset"], args[model_name])

    dataset, dataloader = utils.init_dataloader(args, file_path, batch_size, mode="gan")

    # Build one additional loader from the same dataset (avoid re-loading all images each epoch).
    unlabel_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=2,
        pin_memory=True,
    )

    G = Generator(z_dim)
    DG = MinibatchDiscriminator()
    
    G = torch.nn.DataParallel(G).cuda()
    DG = torch.nn.DataParallel(DG).cuda()

    dg_optimizer = torch.optim.Adam(DG.parameters(), lr=lr, betas=(0.5, 0.999))
    g_optimizer = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))

    entropy = HLoss()
    

    step = 0

    for epoch in range(epochs):
        start = time.time()
        unlabel_iter1 = iter(unlabel_loader)
        unlabel_iter2 = iter(unlabel_loader)

        for i, imgs in enumerate(dataloader):
            current_iter = epoch * len(dataloader) + i + 1

            step += 1
            imgs = imgs.cuda()
            bs = imgs.size(0)
            try:
                x_unlabel = next(unlabel_iter1)
            except StopIteration:
                unlabel_iter1 = iter(unlabel_loader)
                x_unlabel = next(unlabel_iter1)

            try:
                x_unlabel2 = next(unlabel_iter2)
            except StopIteration:
                unlabel_iter2 = iter(unlabel_loader)
                x_unlabel2 = next(unlabel_iter2)

            x_unlabel = x_unlabel.cuda()
            x_unlabel2 = x_unlabel2.cuda()
            
            freeze(G)
            unfreeze(DG)

            z = torch.randn(bs, z_dim).cuda()
            f_imgs = G(z)

            y_prob = T(imgs)[-1]
            y = torch.argmax(y_prob, dim=1).view(-1)
            

            _, output_label = DG(imgs)
            _, output_unlabel = DG(x_unlabel)
            _, output_fake =  DG(f_imgs)

            loss_lab = softXEnt(output_label, y_prob)
            loss_unlab = 0.5*(torch.mean(F.softplus(log_sum_exp(output_unlabel)))-torch.mean(log_sum_exp(output_unlabel))+torch.mean(F.softplus(log_sum_exp(output_fake))))
            dg_loss = loss_lab + loss_unlab
            
            acc = torch.mean((output_label.max(1)[1] == y).float())
            
            
            dg_optimizer.zero_grad()
            dg_loss.backward()
            dg_optimizer.step()

            writer.add_scalar('loss_label_batch', loss_lab, current_iter)
            writer.add_scalar('loss_unlabel_batch', loss_unlab, current_iter)
            writer.add_scalar('DG_loss_batch', dg_loss, current_iter)
            writer.add_scalar('Acc_batch', acc, current_iter)

            # train G

            if step % n_critic == 0:
                freeze(DG)
                unfreeze(G)
                z = torch.randn(bs, z_dim).cuda()
                f_imgs = G(z)
                mom_gen, output_fake = DG(f_imgs)
                mom_unlabel, _ = DG(x_unlabel2)

                mom_gen = torch.mean(mom_gen, dim = 0)
                mom_unlabel = torch.mean(mom_unlabel, dim = 0)

                Hloss = entropy(output_fake)
                g_loss = torch.mean((mom_gen - mom_unlabel).abs()) + 1e-4 * Hloss  

                
                g_optimizer.zero_grad()
                g_loss.backward()
                g_optimizer.step()

                writer.add_scalar('G_loss_batch', g_loss, current_iter)

        end = time.time()
        interval = end - start
        
        print("Epoch:%d \tTime:%.2f\tG_loss:%.2f\t train_acc:%.2f" % (epoch, interval, g_loss, acc))

        torch.save({'state_dict':G.state_dict()}, os.path.join(save_model_dir, "improved_celeba_G.tar"))
        torch.save({'state_dict':DG.state_dict()}, os.path.join(save_model_dir, "improved_celeba_D.tar"))

        if (epoch+1) % 10 == 0:
            z = torch.randn(32, z_dim).cuda()
            fake_image = G(z)
            save_tensor_images(fake_image.detach(), os.path.join(save_img_dir, "improved_celeba_img_{}.png".format(epoch)), nrow = 8)

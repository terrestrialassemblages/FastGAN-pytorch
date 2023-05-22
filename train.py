import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision import utils as vutils
import os

import argparse
import random
from tqdm import tqdm

from models import weights_init, Discriminator, Generator
from operation import copy_G_params, load_params
from operation import ImageFolder, InfiniteSamplerWrapper
from diffaug import DiffAugment
policy = 'color,translation'
import lpips
percept = lpips.PerceptualLoss(model='net-lin', net='vgg', use_gpu=True)

#torch.backends.cudnn.benchmark = True

def crop_image_by_part(image, part):
    hw = image.shape[2]//2
    if part==0:
        return image[:,:,:hw,:hw]
    if part==1:
        return image[:,:,:hw,hw:]
    if part==2:
        return image[:,:,hw:,:hw]
    if part==3:
        return image[:,:,hw:,hw:]

def train_d(net, data, label="real"):
    if label=="real":
        part = random.randint(0, 3)
        pred, [rec_all, rec_small, rec_part] = net(data, label, part=part)
        err = F.relu(  torch.rand_like(pred) * 0.2 + 0.8 -  pred).mean() + \
            percept( rec_all, F.interpolate(data, rec_all.shape[2]) ).sum() +\
            percept( rec_small, F.interpolate(data, rec_small.shape[2]) ).sum() +\
            percept( rec_part, F.interpolate(crop_image_by_part(data, part), rec_part.shape[2]) ).sum()
        err.backward()
        return pred.mean().item(), rec_all, rec_small, rec_part
    else:
        pred = net(data, label)
        err = F.relu( torch.rand_like(pred) * 0.2 + 0.8 + pred).mean()
        err.backward()
        return pred.mean().item()

def train(args):

    data_root = args.path
    total_iterations = args.iter
    checkpoint = args.ckpt
    batch_size = args.batch_size
    im_size = args.im_size
    ndf = 64
    ngf = 64
    nz = 256
    nlr = 0.0002
    nbeta1 = 0.5
    use_cuda = True
    multi_gpu = True
    dataloader_workers = args.workers
    current_iteration = args.start_iter
    save_interval = args.save_interval
    saved_model_folder = args.output_path
    saved_image_folder = args.output_path

     # Create the output directory if it doesn't exist
    os.makedirs(saved_model_folder, exist_ok=True)
    os.makedirs(saved_image_folder, exist_ok=True)

    device = torch.device("cpu")
    if use_cuda:
        device = torch.device("cuda:0")

    transform_list = [
            transforms.Resize((int(im_size),int(im_size))),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]
    trans = transforms.Compose(transform_list)

    if 'lmdb' in data_root:
        from operation import MultiResolutionDataset
        dataset = MultiResolutionDataset(data_root, trans, 1024)
    else:
        dataset = ImageFolder(data_root, trans)

    dataloader = iter(DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=dataloader_workers, sampler=InfiniteSamplerWrapper(dataset)
            ))

    netG = Generator(nz, 3, ngf).to(device)
    netD = Discriminator(3, ndf).to(device)

    if multi_gpu and torch.cuda.device_count() > 1:
        netD = nn.DataParallel(netD)
        netG = nn.DataParallel(netG)

    netG.apply(weights_init)
    netD.apply(weights_init)

    if checkpoint:
        load_params(netG, netD, checkpoint)

    optimizerD = optim.Adam(netD.parameters(), lr=nlr, betas=(nbeta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=nlr, betas=(nbeta1, 0.999))

    fixed_noise = torch.randn(8, nz, device=device)

    if not checkpoint:
        netG.train()
    netD.train()

    for iteration in tqdm(range(current_iteration, total_iterations + 1)):

        if not checkpoint:
            real_image = next(dataloader).to(device)
            fake_image = netG(torch.randn(batch_size, nz, device=device))

            netD.zero_grad()

            real_image.requires_grad_()

            D_real, rec_all, rec_small, rec_part = train_d(netD, real_image)

            real_image.requires_grad_()

            D_fake = train_d(netD, DiffAugment(fake_image.detach(), policy=policy), "fake")

            optimizerD.step()

        netD.zero_grad()
        fake_image = netG(torch.randn(batch_size, nz, device=device))

        D_real, rec_all, rec_small, rec_part = train_d(netD, fake_image)

        optimizerG.step()

        if iteration % save_interval == 0 or iteration == total_iterations:
            vutils.save_image(netG(fixed_noise)[0].add(1).mul(0.5), os.path.join(saved_image_folder, 'image_%d.jpg'%iteration), nrow=4)
            torch.save({'g':netG.state_dict(),'d':netD.state_dict()}, os.path.join(saved_model_folder, 'model_%d.pth'%iteration))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='path of specified dataset')
    parser.add_argument('--ckpt', type=str, default='', help='load from previous checkpoints')
    parser.add_argument('--iter', type=int, default=10000, help='number of epochs to train for')
    parser.add_argument('--save_interval', type=int, default=100, help='save every SAVE_INTERVAL iterations')
    parser.add_argument('--im_size', type=int, default=1024, help='image size')
    parser.add_argument('--start_iter', type=int, default=1, help='starting epoch count')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size, default=4')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers, default=4')
    parser.add_argument('--output_path', type=str, default='./', help='Output path for saving results')
    parser.add_argument('--name', type=str, default='test', help='name of the project')
    args = parser.parse_args()

    train(args)

import argparse
import os
import numpy as np
from tqdm import tqdm
import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim


from models.BigGAN import Generator, Discriminator, G_D
from gan_trainer.pretraining import classifier_pretraining
from gan_trainer import gan_utils
from gan_trainer import train_fns

from data.cifarloader import CIFAR10Loader
from models.resnet import ResNet, BasicBlock
from modules.module import feat2prob, target_distribution 
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from utils.util import cluster_acc
from torch.nn import Parameter


@torch.no_grad()
def test(model, test_loader, args, tsne=False):
    model.eval()
    preds=np.array([])
    targets=np.array([])
    feats = np.zeros((len(test_loader.dataset), args.n_classes))
    device = next(model.parameters()).device
    for batch_idx, (x, label, idx) in enumerate(tqdm(test_loader)):
        x, label = x.to(device), label.to(device)
        feat = model(x)
        prob = feat2prob(feat, model.center)
        _, pred = prob.max(1)
        targets=np.append(targets, label.cpu().numpy())
        preds=np.append(preds, pred.cpu().numpy())
        idx = idx.data.cpu().numpy()
        feats[idx, :] = feat.cpu().detach().numpy()
    acc, nmi, ari = cluster_acc(targets.astype(int), preds.astype(int)), nmi_score(targets, preds), ari_score(targets, preds)
    print('Test acc {:.4f}, nmi {:.4f}, ari {:.4f}'.format(acc, nmi, ari))

    
    return acc, nmi, ari

# Argument parser setup
parser = argparse.ArgumentParser(description='Generative Pseudo-label Refinement for Unsupervised Domain Adaptation',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Data loading parameters
parser.add_argument('--data_path', type=str, default='./datasets')
parser.add_argument('--img_size', type=int, default=32)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--verbose', type=str, default=False, help='Verbose mode')

# GPU
parser.add_argument('--device', type=str, default='cpu', choices=['cuda', 'cpu'])

# Number of classes
parser.add_argument('--n_classes', type=int, default=10)

# Classifier pretraining parameters
parser.add_argument('--n_epochs_cls_pretraining', type=int, default=1)
parser.add_argument('--lr_cls_pretraining', type=float, default=0.5)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=1e-4)

# GAN pretraining parameters
parser.add_argument('--latent_dim', type=int, default=128)
parser.add_argument('--lr_d_pretraining', type=float, default=1e-4)
parser.add_argument('--lr_g_pretraining', type=float, default=1e-4)
parser.add_argument('--n_epochs_gan_pretraining', type=int, default=1)

# Training parameters
parser.add_argument('--lr_cls_training', type=float, default=1e-4)
parser.add_argument('--lr_d_training', type=float, default=1e-4)
parser.add_argument('--lr_g_training', type=float, default=1e-4)
parser.add_argument('--n_epochs_training', type=int, default=0)

# Paths
parser.add_argument('--results_path', type=str, default='./results')
parser.add_argument('--pretraining_path', type=str, default='./results/pretraining')
parser.add_argument('--training_path', type=str, default='./results/training')
parser.add_argument('--cls_pretraining_path', type=str, default='./pretrained/resnet18.pth')

args = parser.parse_args()
args.device = torch.device("cuda" if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')

# Updating paths based on args
args.img_training_path = os.path.join(args.training_path, 'images')
args.models_training_path = os.path.join(args.training_path, 'models')
os.makedirs(args.img_training_path, exist_ok=True)


# Parameters (Default for now)
args.n_classes = 5
batch_size = 50
num_D_steps = 4
D_batch_size = batch_size*num_D_steps
G_batch_size = batch_size # max ( args.G_batch_size , batch_size)



# --------------------
#   Data loading
# --------------------
train_loader = CIFAR10Loader(root=args.data_path, batch_size=D_batch_size, split='train', aug='twice', shuffle=True, target_list=range(0, 10))
eval_loader = CIFAR10Loader(root=args.data_path, batch_size=D_batch_size, split='train', aug=None, shuffle=False, target_list=range(0, 10))
# --------------------

# Classifier pretraining 

classifier = classifier_pretraining(args, train_loader, eval_loader)
# init_acc, init_nmi, init_ari = test(classifier, eval_loader, args)

# print('Init ACC {:.4f}, NMI {:.4f}, ARI {:.4f}'.format(init_acc, init_nmi, init_ari))


# GAN pretraining on target data annotated by classifier
G = Generator(n_classes=args.n_classes, dim_z=args.latent_dim, resolution= args.img_size).to(args.device)
D = Discriminator(n_classes=args.n_classes, resolution= args.img_size).to(args.device)

GD = G_D(G, D)


  # Prepare noise and randomly sampled label arrays
  # Allow for different batch sizes in G

z_, y_ = gan_utils.prepare_z_y(G_batch_size=G_batch_size, dim_z = args.latent_dim, nclasses= args.n_classes,   device=args.device)

 # Prepare a fixed z & y to see individual sample evolution throghout training
fixed_z, fixed_y = gan_utils.prepare_z_y(G_batch_size=G_batch_size, dim_z = args.latent_dim, nclasses= args.n_classes,   device=args.device)

fixed_z.sample_()
fixed_y.sample_()

train = train_fns.GAN_training_function(G, D, GD, z_, y_, batch_size, num_D_steps, num_D_accumulations=1, num_G_accumulations=1)

DEBUG = True

for epoch in range(args.n_epochs_gan_pretraining):
    if DEBUG: break
    for i, ((images, _),  targets, _) in enumerate(tqdm(train_loader)):
        x = images.to(args.device)
        y = (targets - 5).to(args.device)

        G.train()
        D.train()

        metrics = train(x, y)

print('Finished Training GAN')
print('\n')
G.eval()
print('Generating sample image\n')
with torch.no_grad():
    fixed_Gz = G(fixed_z, G.shared(fixed_y))
print(fixed_Gz.shape)
image_filename = '%s/fixed_sample.jpg' % (args.img_training_path)

print(fixed_Gz)
print(fixed_Gz.float())

torchvision.utils.save_image(torch.tensor(fixed_Gz.float().cpu()), image_filename,
                             nrow=int(fixed_Gz.shape[0] **0.5), normalize=True)
# --------------------
#     Final Model
# --------------------

acc, nmi, ari = test(classifier, eval_loader, args)
# print('Init ACC {:.4f}, NMI {:.4f}, ARI {:.4f}'.format(init_acc, init_nmi, init_ari))
print('Final ACC {:.4f}, NMI {:.4f}, ARI {:.4f}'.format(acc, nmi, ari))




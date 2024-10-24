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
from utils.util import cluster_acc, renormalize_to_standard
from torch.nn import Parameter
import matplotlib.pyplot as plt

from data.simpleCIFAR import get_simple_data_loader

@torch.no_grad()
def test(model, test_loader, args, tsne=False):
    model.eval()
    preds=np.array([])
    targets=np.array([])
    feats = np.zeros((len(test_loader.dataset), args.n_unlabeled_classes))
    device = next(model.parameters()).device
    for batch_idx, (x, label, idx) in enumerate(tqdm(test_loader)):
        x, label = x.to(device), label.to(device)
        x = renormalize_to_standard(x)
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
parser.add_argument('--n_labeled_classes', default=5, type=int)
parser.add_argument('--n_unlabeled_classes', type=int, default=5)


# GAN pretraining parameters
parser.add_argument('--pretrained_gan', type=bool, default=True)
parser.add_argument('--latent_dim', type=int, default=128)
parser.add_argument('--n_epochs_gan_pretraining', type=int, default=1)

# Training parameters
parser.add_argument('--lr_cls_training', type=float, default=1e-4)
parser.add_argument('--n_epochs_training', type=int, default=0)
parser.add_argument('--num_D_steps', type=int, default=4)
parser.add_argument('--num_G_steps', type=int, default=1)
# Paths
parser.add_argument('--results_path', type=str, default='./results')
parser.add_argument('--pretraining_path', type=str, default='./results/pretraining')
parser.add_argument('--models_pretraining_path', type=str, default='')
parser.add_argument('--training_path', type=str, default='./results/training')
parser.add_argument('--cls_pretraining_path', type=str, default='./pretrained/resnet18.pth')

args = parser.parse_args()
args.device = torch.device("cuda" if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')

print(args.device)
# Updating paths based on args
args.img_training_path = os.path.join(args.training_path, 'images')
args.models_training_path = os.path.join(args.training_path, 'models')
os.makedirs(args.img_training_path, exist_ok=True)
os.makedirs(args.models_training_path, exist_ok=True)

args.img_pretraining_path = os.path.join(args.pretraining_path, 'images')
os.makedirs(args.img_pretraining_path, exist_ok=True)

if args.models_pretraining_path != '':
    args.models_pretraining_path = os.path.join(args.pretraining_path, 'models')
    os.makedirs(args.models_pretraining_path, exist_ok=True)


# Parameters (Default for now)
D_batch_size = args.batch_size*args.num_D_steps
G_batch_size = args.batch_size # max ( args.G_batch_size , batch_size)


# --------------------
#   Data loading
# --------------------
train_loader = CIFAR10Loader(root=args.data_path, batch_size=D_batch_size, split='train', aug='gan', shuffle=True, target_list=range(5, 10))
eval_loader = CIFAR10Loader(root=args.data_path, batch_size=D_batch_size, split='train', aug=None, shuffle=False, target_list=range(0, 5))
# train_loader = get_simple_data_loader()
# --------------------

# Classifier pretraining 

classifier = classifier_pretraining(args)
init_acc, init_nmi, init_ari = test(classifier, train_loader, args)

print('Init ACC {:.4f}, NMI {:.4f}, ARI {:.4f}'.format(init_acc, init_nmi, init_ari))


# GAN pretraining on target data annotated by classifier
G = Generator(n_classes=args.n_unlabeled_classes, dim_z=args.latent_dim, resolution= args.img_size).to(args.device)
D = Discriminator(n_classes=args.n_unlabeled_classes, resolution= args.img_size).to(args.device)

GD = G_D(G, D)

print(G)
print(D)
print(GD)


  # Prepare noise and randomly sampled label arrays
  # Allow for different batch sizes in G

z_, y_ = gan_utils.prepare_z_y(G_batch_size=G_batch_size, dim_z = args.latent_dim, nclasses= args.n_unlabeled_classes,   device=args.device)

 # Prepare a fixed z & y to see individual sample evolution throghout training
fixed_z, fixed_y = gan_utils.prepare_z_y(G_batch_size=G_batch_size, dim_z = args.latent_dim, nclasses= args.n_unlabeled_classes,   device=args.device)

fixed_z.sample_()
# fixed_y.sample_()

# Create a tensor where each class (0 to 4) appears 10 times, repeated serially
fixed_y = torch.tensor([i for i in range(args.n_unlabeled_classes) for _ in range(G_batch_size // args.n_unlabeled_classes)])

# Ensure the tensor is on the correct device
fixed_y = fixed_y.to(args.device)

# Initialize tracking variables
pretrain_itr = 1
G_losses = []
D_losses_real = []
D_losses_fake = []

# Initialize tracking variables for epoch-wise losses
epoch_G_losses = []
epoch_D_losses_real = []
epoch_D_losses_fake = []

train = train_fns.GAN_training_function(G, D, GD, z_, y_, args.batch_size, args.num_D_steps, num_D_accumulations=1, num_G_accumulations=1)

# Load pre-trained GAN models if available
# Load pre-trained GAN models if available
if args.pretrained_gan and os.path.exists(os.path.join(args.models_pretraining_path, 'G.pth')):
    G = torch.load(os.path.join(args.models_pretraining_path, 'G.pth')).to(args.device)
    D = torch.load(os.path.join(args.models_pretraining_path, 'D.pth')).to(args.device)
    print("Loaded pre-trained GAN models.")
# if args.pretrained_gan and os.path.exists(args.pretrained_G) and os.path.exists(args.pretrained_D):
#     G = torch.load(args.pretrained_G).to(args.device)
#     D = torch.load(args.pretrained_D).to(args.device)
#     print("Loaded pre-trained GAN models.")
else:
    G = Generator(n_classes=args.n_unlabeled_classes, dim_z=args.latent_dim, resolution=args.img_size).to(args.device)
    D = Discriminator(n_classes=args.n_unlabeled_classes, resolution=args.img_size).to(args.device)
    GD = G_D(G, D)
    print("Training GAN models from scratch.")

    # Prepare noise and labels for GAN
    z_, y_ = gan_utils.prepare_z_y(G_batch_size=args.batch_size, dim_z=args.latent_dim, nclasses=args.n_unlabeled_classes, device=args.device)
    fixed_z, fixed_y = gan_utils.prepare_z_y(G_batch_size=args.batch_size, dim_z=args.latent_dim, nclasses=args.n_unlabeled_classes, device=args.device)
    fixed_z.sample_()
    fixed_y = torch.tensor([i for i in range(args.n_unlabeled_classes) for _ in range(args.batch_size // args.n_unlabeled_classes)]).to(args.device)

    # GAN training function
    train = train_fns.GAN_training_function(G, D, GD, z_, y_, args.batch_size, args.num_D_steps, num_D_accumulations=1, num_G_accumulations=1)

    # Training loop
    for epoch in range(args.n_epochs_gan_pretraining):
        G_loss_epoch = 0
        D_loss_real_epoch = 0
        D_loss_fake_epoch = 0
        num_batches = 0

        for i, (images, targets, idx) in enumerate(tqdm(train_loader)):
            x = images.to(args.device)
            y = (targets - 5).to(args.device)
            x_classifier = renormalize_to_standard(x)
            feat = classifier(x_classifier)
            prob = feat2prob(feat, classifier.center)
            _, y = prob.max(1)
            y = y.to(args.device)

            # Train GAN
            metrics = train(x, y)
            G_losses.append(metrics['G_loss'])
            D_losses_real.append(metrics['D_loss_real'])
            D_losses_fake.append(metrics['D_loss_fake'])

            # print(f"Iter {pretrain_itr}, G_loss: {metrics['G_loss']:.4f}, D_loss_real: {metrics['D_loss_real']:.4f}, D_loss_fake: {metrics['D_loss_fake']:.4f}")
            pretrain_itr += 1
            # Accumulate epoch losses
            G_loss_epoch += metrics['G_loss']
            D_loss_real_epoch += metrics['D_loss_real']
            D_loss_fake_epoch += metrics['D_loss_fake']

            num_batches += 1

        # Calculate average losses for the epoch
        G_loss_epoch_avg = G_loss_epoch / num_batches
        D_loss_real_epoch_avg = D_loss_real_epoch / num_batches
        D_loss_fake_epoch_avg = D_loss_fake_epoch / num_batches

        # Append to epoch-wise loss lists
        epoch_G_losses.append(G_loss_epoch_avg)
        epoch_D_losses_real.append(D_loss_real_epoch_avg)
        epoch_D_losses_fake.append(D_loss_fake_epoch_avg)

        # Save models after each epoch to pretraining path
        torch.save(G, os.path.join(args.models_pretraining_path, 'G.pth'))
        torch.save(D, os.path.join(args.models_pretraining_path, 'D.pth'))
        print(f"Epoch {epoch + 1} completed. GAN models saved. G_loss: {G_loss_epoch_avg:.4f}, D_loss_real: {D_loss_real_epoch_avg:.4f}, D_loss_fake: {D_loss_fake_epoch_avg:.4f}")

    # Save losses in pretraining path
    loss_data = {'G_losses': G_losses, 'D_losses_real': D_losses_real, 'D_losses_fake': D_losses_fake}
    np.save(os.path.join(args.pretraining_path, 'gan_pretraining_losses.npy'), loss_data)
    print("Loss data saved.")

    # Plot and save the loss curves
    plt.figure()
    plt.plot(G_losses, label="G_loss")
    plt.plot(D_losses_real, label="D_loss_real")
    plt.plot(D_losses_fake, label="D_loss_fake")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("GAN Pretraining Loss")
    plt.savefig(os.path.join(args.pretraining_path, 'gan_pretraining_loss_plot.png'))

    # Save epoch-wise loss data
    epoch_loss_data = {'epoch_G_losses': epoch_G_losses, 'epoch_D_losses_real': epoch_D_losses_real, 'epoch_D_losses_fake': epoch_D_losses_fake}
    np.save(os.path.join(args.pretraining_path, 'gan_pretraining_epoch_losses.npy'), epoch_loss_data)
    print("Epoch-wise loss data saved.")

    # Plot and save the epoch-wise loss curves
    plt.figure()
    plt.plot(epoch_G_losses, label="G_loss")
    plt.plot(epoch_D_losses_real, label="D_loss_real")
    plt.plot(epoch_D_losses_fake, label="D_loss_fake")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("GAN Pretraining Loss (Epoch-wise)")
    plt.savefig(os.path.join(args.pretraining_path, 'gan_pretraining_epoch_loss_plot.png'))


# Generate sample image
G.eval()
with torch.no_grad():
    fixed_Gz = nn.parallel.data_parallel(G, (fixed_z, G.shared(fixed_y)))
image_filename = os.path.join(args.img_pretraining_path, 'fixed_sample.jpg')
torchvision.utils.save_image(fixed_Gz.float().cpu(), image_filename, nrow=int(fixed_Gz.shape[0] ** 0.5), normalize=True)
print(f"Sample images saved at {image_filename}.")

#     Final Model
# --------------------

# acc, nmi, ari = test(classifier, eval_loader, args)
# print('Init ACC {:.4f}, NMI {:.4f}, ARI {:.4f}'.format(init_acc, init_nmi, init_ari))
# print('Final ACC {:.4f}, NMI {:.4f}, ARI {:.4f}'.format(acc, nmi, ari))




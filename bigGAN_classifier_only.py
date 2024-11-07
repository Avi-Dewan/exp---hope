import argparse
import os
import numpy as np
from tqdm import tqdm
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim


from models.BigGAN import Generator, Discriminator, G_D
from gan_trainer.pretraining import classifier_pretraining, gan_pretraining
from gan_trainer import gan_utils
from gan_trainer import train_fns

from data.cifarloader import CIFAR10Loader
from models.resnet import ResNet, BasicBlock
from modules.module import feat2prob, target_distribution 
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from utils.util import cluster_acc
from utils import ramps
from data.utils import renormalize_to_standard, create_two_views
from torch.nn import Parameter
import matplotlib.pyplot as plt

# --------------------
# NCD metrics 
# --------------------

@torch.no_grad()
def test(model, test_loader, args, tsne=False):
    model.eval()
    preds=np.array([])
    targets=np.array([])
    feats = np.zeros((len(test_loader.dataset), args.n_unlabeled_classes))
    device = next(model.parameters()).device
    for batch_idx, (x, label, idx) in enumerate(tqdm(test_loader)):
        x, label = x.to(device), label.to(device)
        # x = renormalize_to_standard(x)
        feat = model(x)
        prob = feat2prob(feat, model.center)
        _, pred = prob.max(1)
        targets=np.append(targets, label.cpu().numpy())
        preds=np.append(preds, pred.cpu().numpy())
        idx = idx.data.cpu().numpy()
        feats[idx, :] = feat.cpu().detach().numpy()
    acc, nmi, ari = cluster_acc(targets.astype(int), preds.astype(int)), nmi_score(targets, preds), ari_score(targets, preds)
    
    return acc, nmi, ari


# --------------------
# Argument parser setup
# --------------------

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
parser.add_argument('--pretrain_save_interval', type=int, default=50)

# Training parameters
parser.add_argument('--cls_lr_training', type=float, default=0.05)
parser.add_argument('--cls_momentum', type=float, default=0.9)
parser.add_argument('--cls_weight_decay', type=float, default=1e-4)
parser.add_argument('--rampup_length', default=5, type=int)
parser.add_argument('--rampup_coefficient', type=float, default=10.0)
parser.add_argument('--n_epochs_training', type=int, default=1)
parser.add_argument('--num_D_steps', type=int, default=4)
parser.add_argument('--num_G_steps', type=int, default=1)
parser.add_argument('--num_C_steps', type=int, default=4)
parser.add_argument('--save_interval', type=int, default=50)

# Paths
parser.add_argument('--results_path', type=str, default='./results')
parser.add_argument('--pretraining_path', type=str, default='./results/pretraining/models')
parser.add_argument('--training_path', type=str, default='./results/training')
parser.add_argument('--cls_pretraining_path', type=str, default='./pretrained/resnet18.pth')

args = parser.parse_args()
args.device = torch.device("cuda" if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')

# Updating paths based on args
args.img_training_path = os.path.join(args.training_path, 'images')
args.models_training_path = os.path.join(args.training_path, 'models')
os.makedirs(args.img_training_path, exist_ok=True)
os.makedirs(args.models_training_path, exist_ok=True)



# --------------------
#   Data loading
# --------------------
# Generator and Discriminator batch size
D_batch_size = args.batch_size*args.num_D_steps
G_batch_size = args.batch_size # max ( args.G_batch_size , batch_size)

train_loader = CIFAR10Loader(root=args.data_path, batch_size=D_batch_size, split='train', aug='gan', shuffle=True, target_list=range(5, 10))
eval_loader = CIFAR10Loader(root=args.data_path, batch_size=D_batch_size, split='train', aug=None, shuffle=False, target_list=range(5, 10))
# --------------------


# --------------------
# Classifier pretraining 
# --------------------

classifier = classifier_pretraining(args)
# init_acc, init_nmi, init_ari = test(classifier, eval_loader, args)

# print('Init ACC {:.4f}, NMI {:.4f}, ARI {:.4f}'.format(init_acc, init_nmi, init_ari))
# # --------------------



# --------------------
# GAN pretraining
# --------------------

# Prepare noise and randomly sampled label arrays
# Allow for different batch sizes in G
z_, y_ = gan_utils.prepare_z_y(G_batch_size=G_batch_size, dim_z = args.latent_dim, nclasses= args.n_unlabeled_classes,   device=args.device)

# Prepare a fixed z & y to see individual sample evolution throghout training
fixed_z, fixed_y = gan_utils.prepare_z_y(G_batch_size=G_batch_size, dim_z = args.latent_dim, nclasses= args.n_unlabeled_classes,   device=args.device)

fixed_z.sample_()
# fixed_y.sample_()

# Create a tensor where each class (0 to 4) appears 10 times, repeated serially
fixed_y = torch.tensor([i for i in range(args.n_unlabeled_classes) for _ in range(args.batch_size // args.n_unlabeled_classes)]).to(args.device)

G, D = gan_pretraining(classifier, train_loader, z_, y_, fixed_z, fixed_y, args)

# Generate and save First sample image
G.eval()
with torch.no_grad():
    if args.device == torch.device('cpu'):
        fixed_Gz = G(fixed_z, G.shared(fixed_y))  # No data parallelism for CPU or single GPU
    else:
        fixed_Gz = nn.parallel.data_parallel(G, (fixed_z, G.shared(fixed_y)))  # For multiple GPUs

# --------------------




# --------------------
# Main Training
# --------------------

GD = G_D(G, D)

train = train_fns.GAN_training_function(G, D, GD, z_, y_, args.batch_size, args.num_D_steps, num_D_accumulations=1, num_G_accumulations=1)

cls_optimizer = optim.SGD(classifier.parameters(), lr=args.cls_lr_training, momentum=args.cls_momentum, weight_decay=args.cls_weight_decay)
CE_loss = nn.CrossEntropyLoss().to(args.device)

print("Starting Main Training Loop...")

# epoch_CE_losses = []
epoch_Consistency_losses = []
epoch_C_losses = []

# Initialize lists to store classfier metrics and their corresponding evaluation epochs
eval_epochs, acc_list, nmi_list, ari_list = [], [], [], []

# Only train the classifier
# w = 0

for epoch in range(args.n_epochs_training):
    classifier.train()
    G.eval()
    cls_optimizer.zero_grad()
    # w = args.rampup_coefficient * ramps.sigmoid_rampup(epoch, args.rampup_length)

    z_.sample_()
    y_.sample_()
    with torch.no_grad():
        if args.device == torch.device('cpu'):
            fake_images = G(z_, G.shared(y_))  # No data parallelism for CPU or single GPU
        else:
            fake_images = nn.parallel.data_parallel(G, (z_, G.shared(y_)))  # For multiple GPUs

        # fake_images = nn.parallel.data_parallel(G, (z_, G.shared(y_)))
        
    # fake_labels = y_

    # # fake_images = torch.tensor(fake_images).to(args.device)
    # y = torch.tensor(y_).to(args.device)

    # x = renormalize_to_standard(fake_images).to(args.device)
    # # y = y_.clone().detach().to(args.device)

    # feat = classifier(x)
    # prob = feat2prob(feat, classifier.center)
    # cross_entropy_loss = CE_loss(prob.log(), y)

    x, x_bar = create_two_views(fake_images)
    x, x_bar = x.to(args.device), x_bar.to(args.device)
    feat = classifier(x)
    feat_bar = classifier(x_bar)
    prob = feat2prob(feat, classifier.center)
    prob_bar = feat2prob(feat_bar, classifier.center)

    consistency_loss = F.mse_loss(prob, prob_bar)

    # cls_loss = cross_entropy_loss + w*consistency_loss
    # cls_loss = cross_entropy_loss + consistency_loss
    cls_loss = consistency_loss
    cls_loss.backward()
    cls_optimizer.step()


    epoch_C_losses.append(float(cls_loss.item()))
    # epoch_CE_losses.append(float(cross_entropy_loss.item()))
    epoch_Consistency_losses.append(float(consistency_loss.item()))

    acc, nmi, ari = test(classifier, eval_loader, args)
    eval_epochs.append(epoch)
    acc_list.append(acc)
    nmi_list.append(nmi)
    ari_list.append(ari)

    # print(f"Epoch {epoch + 1}/{args.n_epochs_training}:\t  CE_loss: {float(cross_entropy_loss.item()):.4f}, Consistency_loss: {float(consistency_loss.item()):.4f}, C_loss: {float(cls_loss.item()):.4f}")

    print(f"Epoch {epoch + 1}/{args.n_epochs_training}:\t Consistency_loss: {float(consistency_loss.item()):.4f}, C_loss: {float(cls_loss.item()):.4f}")

    
    print('Test acc {:.4f}, nmi {:.4f}, ari {:.4f}'.format(acc, nmi, ari))


       
print("Training Done.\n Saving losses and metrics data...")





# --------------------------
#   Loss and metrics plot
# --------------------------



# Plot and save the epoch-wise loss curves
plt.figure()
plt.plot(epoch_C_losses, label="C_loss")
# plt.plot(epoch_CE_losses, label="CE_loss")
plt.plot(epoch_Consistency_losses, label="Consistency_loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training Loss (Epoch-wise)")
plt.savefig(os.path.join(args.training_path, 'training_epoch_loss_plot.png'))


# Plot and save the evaluation metrics curves
plt.figure()
plt.plot(eval_epochs, acc_list, label="Accuracy", marker='o')
plt.plot(eval_epochs, nmi_list, label="NMI", marker='o')
plt.plot(eval_epochs, ari_list, label="ARI", marker='o')
plt.xlabel("Epoch")
plt.ylabel("Metric")
plt.legend()
plt.title("Classifier Evaluation Metrics (Periodically)")
plt.savefig(os.path.join(args.training_path, 'classifier_evaluation_metrics_plot.png'))

# --------------------

# --------------------
#     Final Model
# --------------------

acc, nmi, ari = test(classifier, eval_loader, args)
# print('Init ACC {:.4f}, NMI {:.4f}, ARI {:.4f}'.format(init_acc, init_nmi, init_ari))
print('Final ACC {:.4f}, NMI {:.4f}, ARI {:.4f}'.format(acc, nmi, ari))




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
parser.add_argument('--n_epochs_training', type=int, default=0)
parser.add_argument('--num_D_steps', type=int, default=4)
parser.add_argument('--num_G_steps', type=int, default=1)
parser.add_argument('--num_C_steps', type=int, default=4)
parser.add_argument('--save_interval', type=int, default=50)

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
init_acc, init_nmi, init_ari = test(classifier, eval_loader, args)

print('Init ACC {:.4f}, NMI {:.4f}, ARI {:.4f}'.format(init_acc, init_nmi, init_ari))
# --------------------



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
    fixed_Gz = nn.parallel.data_parallel(G, (fixed_z, G.shared(fixed_y)))
image_filename = os.path.join(args.img_training_path, f'fixed_sample0.jpg')
torchvision.utils.save_image(fixed_Gz.float().cpu(), image_filename, nrow=int(fixed_Gz.shape[0] ** 0.5), normalize=True)
print(f"Sample images saved at {image_filename}.")
# --------------------




# --------------------
# Main Training
# --------------------

GD = G_D(G, D)

train = train_fns.GAN_training_function(G, D, GD, z_, y_, args.batch_size, args.num_D_steps, num_D_accumulations=1, num_G_accumulations=1)

cls_optimizer = optim.SGD(classifier.parameters(), lr=args.cls_lr_training, momentum=args.cls_momentum, weight_decay=args.cls_weight_decay)
cls_loss = nn.CrossEntropyLoss().to(args.device)

print("Starting Main Training Loop...")

w = 0

# Initialize tracking variables for itertaion wise
pretrain_itr = 1
G_losses = []
D_losses_real = []
D_losses_fake = []

# Initialize tracking variables for epoch-wise losses
epoch_G_losses = []
epoch_D_losses_real = []
epoch_D_losses_fake = []
epoch_C_losses = []

# Initialize lists to store classfier metrics and their corresponding evaluation epochs
eval_epochs, acc_list, nmi_list, ari_list = [], [], [], []

# Training loop
for epoch in range(args.n_epochs_training):
    G_loss_epoch = 0
    D_loss_real_epoch = 0
    D_loss_fake_epoch = 0
    C_loss_epoch = 0
    num_batches = 0

    w = args.rampup_coefficient * ramps.sigmoid_rampup(epoch, args.rampup_length)

    for i, (images, _ , idx) in enumerate(tqdm(train_loader)):

        # Train the classifier: 
        classifier.train()
        G.eval()

        for step_index in range(args.num_C_steps):

            cls_optimizer.zero_grad()

            z_.sample_()
            y_.sample_()
            with torch.no_grad():
                fake_images = nn.parallel.data_parallel(G, (z_, G.shared(y_)))
            fake_labels = y_

            x = renormalize_to_standard(fake_images).to(args.device)
            # y = y_.clone().detach().to(args.device)
            y = torch.tensor(y_).to(args.device)

            feat = classifier(x)
            prob = feat2prob(feat, classifier.center)
            cross_entropy_loss = cls_loss(prob, y)

            # x, x_bar = create_two_views(fake_images)
            # x, x_bar = x.to(args.device), x_bar.to(args.device)
            # feat = classifier(x)
            # feat_bar = classifier(x_bar)
            # prob = feat2prob(feat, classifier.center)
            # prob_bar = feat2prob(feat_bar, classifier.center)

            # consistency_loss = F.mse_loss(prob, prob_bar)

            cls_loss = cross_entropy_loss # + w*consistency_loss
            cls_loss.backward()
            cls_optimizer.step()

            # Accumulate loss per steps, per epoch
            C_loss_epoch += float(cls_loss.item())

        # Train the GAN
        classifier.eval()

        x = images.to(args.device)
        with torch.no_grad():
            x_classifier = renormalize_to_standard(x)
            feat = classifier(x_classifier)
            prob = feat2prob(feat, classifier.center)
            _, y = prob.max(1)

        y = y.to(args.device)

        # GAN training step
        metrics = train(x, y)

        # Track per iteration losses
        G_losses.append(metrics['G_loss'])
        D_losses_real.append(metrics['D_loss_real'])
        D_losses_fake.append(metrics['D_loss_fake'])

        print(f"Iter {pretrain_itr}, G_loss: {metrics['G_loss']:.4f}, D_loss_real: {metrics['D_loss_real']:.4f}, D_loss_fake: {metrics['D_loss_fake']:.4f}")
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
    C_loss_epoch_avg = C_loss_epoch / (num_batches*args.num_C_steps)

    # Append to epoch-wise loss lists
    epoch_G_losses.append(G_loss_epoch_avg)
    epoch_D_losses_real.append(D_loss_real_epoch_avg)
    epoch_D_losses_fake.append(D_loss_fake_epoch_avg)
    epoch_C_losses.append(C_loss_epoch_avg)

    print(f"Epoch {epoch + 1}/{args.n_epochs_training}:   G_loss: {G_loss_epoch_avg:.4f}, D_loss_real: {D_loss_real_epoch_avg:.4f}, D_loss_fake: {D_loss_fake_epoch_avg:.4f}, C_loss: {C_loss_epoch_avg:.4f}")
    
    # Save models at each interval
    if (epoch == args.n_epochs_training-1 or ((epoch+1) % args.save_interval == 0)):
        torch.save(G, os.path.join(args.models_pretraining_path, 'G.pth'))
        torch.save(D, os.path.join(args.models_pretraining_path, 'D.pth'))

        print(f"GAN model saved at {epoch+1}")

        # Generate sample image
        G.eval()
        with torch.no_grad():
            fixed_Gz = nn.parallel.data_parallel(G, (fixed_z, G.shared(fixed_y)))
        image_filename = os.path.join(args.img_training_path, f'fixed_sample{epoch + 1}.jpg')
        torchvision.utils.save_image(fixed_Gz.float().cpu(), image_filename, nrow=int(fixed_Gz.shape[0] ** 0.5), normalize=True)
        print(f"Sample images saved at {image_filename}.")


        acc, nmi, ari = test(classifier, eval_loader, args)
        eval_epochs.append(epoch)
        acc_list.append(acc)
        nmi_list.append(nmi)
        ari_list.append(ari)
        print('Test acc {:.4f}, nmi {:.4f}, ari {:.4f}'.format(acc, nmi, ari))

print("Traing Done.\n Saving losses and metrics data...")
# --------------------




# --------------------------
#   Loss and metrics plot
# --------------------------

# Save losses in training path
loss_data = {'G_losses': G_losses, 'D_losses_real': D_losses_real, 'D_losses_fake': D_losses_fake}
np.save(os.path.join(args.training_path, 'gan_training_iter_losses.npy'), loss_data)
print("Loss data saved.")

# Plot and save the loss curves
plt.figure()
plt.plot(G_losses, label="G_loss")
plt.plot(D_losses_real, label="D_loss_real")
plt.plot(D_losses_fake, label="D_loss_fake")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.legend()
plt.title("GAN Training Loss (Iter-wise)")
plt.savefig(os.path.join(args.training_path, 'gan_training_iter_loss_plot.png'))

# Save epoch-wise loss data
epoch_loss_data = {'epoch_G_losses': epoch_G_losses, 'epoch_D_losses_real': epoch_D_losses_real, 'epoch_D_losses_fake': epoch_D_losses_fake, 'epoch_C_losses': epoch_C_losses}
np.save(os.path.join(args.training_path, 'gan_training_epoch_losses.npy'), epoch_loss_data)
print("Epoch-wise loss data saved.")

# Plot and save the epoch-wise loss curves
plt.figure()
plt.plot(epoch_G_losses, label="G_loss")
plt.plot(epoch_D_losses_real, label="D_loss_real")
plt.plot(epoch_D_losses_fake, label="D_loss_fake")
plt.plot(epoch_C_losses, label="C_loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("GAN Training Loss (Epoch-wise)")
plt.savefig(os.path.join(args.training_path, 'gan_training_epoch_loss_plot.png'))

# Save metrics along with the evaluation epochs
epoch_metric_data = {
    'eval_epochs': eval_epochs,
    'epoch_acc': acc_list,
    'epoch_nmi': nmi_list,
    'epoch_ari': ari_list
}
np.save(os.path.join(args.training_path, 'gan_training_eval_metrics.npy'), epoch_metric_data)
print("Evaluation metrics data saved.")

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
print('Init ACC {:.4f}, NMI {:.4f}, ARI {:.4f}'.format(init_acc, init_nmi, init_ari))
print('Final ACC {:.4f}, NMI {:.4f}, ARI {:.4f}'.format(acc, nmi, ari))




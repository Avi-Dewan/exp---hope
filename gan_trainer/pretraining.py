"""
Pretraining functions for classifier and GAN
"""
import os

import numpy as np

import torch
from tqdm import tqdm
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.nn.parameter import Parameter

from torchvision.utils import save_image

from gan_trainer.training_step import classifier_train_step, generator_train_step, discriminator_train_step
from modules.module import feat2prob, target_distribution 

from models.resnet import ResNet, BasicBlock
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from utils.util import cluster_acc, Identity, AverageMeter, seed_torch, str2bool


def getPsedoLabels(model, train_loader, args):
    model.eval()
    pseudoLabels=np.array([])
    targets=np.array([])

    device = next(model.parameters()).device
    for batch_idx, ((x, _), label, _) in enumerate(tqdm(train_loader)):
        x = x.to(device)
        feat = model(x)
        prob = feat2prob(feat, model.center)
        _, pseudoLabel = prob.max(1)
        pseudoLabels=np.append(pseudoLabels, pseudoLabel.cpu().numpy())
        targets=np.append(targets, label.cpu().numpy())

    acc, nmi, ari = cluster_acc(targets.astype(int), pseudoLabels.astype(int)), nmi_score(targets, pseudoLabels), ari_score(targets, pseudoLabels)
    print('PseudoLabel acc {:.4f}, nmi {:.4f}, ari {:.4f}'.format(acc, nmi, ari))

    return pseudoLabels

def classifier_pretraining(args):
    # Classifier pretraining on source data
    model_dict = torch.load(args.cls_pretraining_path, map_location=args.device, weights_only=False)
    model = ResNet(BasicBlock, [2,2,2,2], args.n_unlabeled_classes).to(args.device)
    model.load_state_dict(model_dict['state_dict'], strict=False)
    model.center = Parameter(model_dict['center'])
    return model


# def gan_pretraining(generator, discriminator, classifier, loader_train,
#                     lr_g, lr_d, latent_dim, n_classes, n_epochs,
#                     img_size, results_path):
#     """
#     Args:
#         generator (TYPE)
#         discriminator (TYPE)
#         classifier (TYPE)
#         loader_train (TYPE)
#         lr_g (TYPE)
#         lr_d (TYPE)
#         latent_dim (TYPE)
#         n_classes (TYPE)
#         n_epochs (TYPE)
#         img_size (TYPE)
#         results_path (TYPE)

#     Returns:
#         Pretrained generator and discriminator
#     """
#     img_pretraining_path = ''.join([results_path, '/images'])
#     models_pretraining_path = ''.join([results_path, '/gan_models'])

#     g_pretrained = ''.join([results_path, '/generator_pretrained.pth'])
#     d_pretrained = ''.join([results_path, '/discriminator_pretrained.pth'])

#     device = next(classifier.parameters()).device

#     os.makedirs(img_pretraining_path, exist_ok=True)
#     os.makedirs(models_pretraining_path, exist_ok=True)

#     loaded_gen = False
#     loaded_dis = False

#     if os.path.isfile(g_pretrained):
#         generator.load_state_dict(torch.load(g_pretrained))
#         print('loaded existing generator')
#         loaded_gen = True

#     if os.path.isfile(d_pretrained):
#         discriminator.load_state_dict(torch.load(d_pretrained))
#         print('loaded existing discriminator')
#         loaded_dis = True

    

#     if not(loaded_gen and loaded_dis):
#         print('Starting Pre Training GAN')
#         for epoch in range(n_epochs):
#             print(f'Starting epoch {epoch}/{n_epochs}...', end=' ')
#             g_loss_list = []
#             d_loss_list = []
#             for i, ((images, _), _, _) in enumerate(tqdm(loader_train)):

#                 # real_images = Variable(images).to(device)
#                 # _, labels = torch.max(classifier(real_images), dim=1)

#                 real_images = Variable(images).to(device)
#                 feat = classifier(real_images)
#                 prob = feat2prob(feat, classifier.center)
#                 _, labels = prob.max(1)

#                 generator.train()

#                 d_loss = discriminator_train_step(discriminator, generator, d_optimizer, criterion_gan,
#                                                   real_images, labels, latent_dim, n_classes)
#                 d_loss_list.append(d_loss)

#                 g_loss = generator_train_step(discriminator, generator, g_optimizer, criterion_gan,
#                                               loader_train.batch_size, latent_dim, n_classes=n_classes)
#                 g_loss_list.append(g_loss)

#             generator.eval()

#             latent_space = Variable(torch.randn(n_classes, latent_dim)).to(device)
#             gen_labels = Variable(torch.LongTensor(np.arange(n_classes))).to(device)

#             gen_imgs = generator(latent_space, gen_labels).view(-1, 3, img_size, img_size)

#             if epoch == n_epochs - 1:

#                 save_image(gen_imgs.data, img_pretraining_path + f'/epoch_{epoch:02d}.png', nrow=n_classes, normalize=True)
#                 torch.save(generator.state_dict(), models_pretraining_path + f'/{epoch:02d}_gen.pth')
#                 torch.save(discriminator.state_dict(), models_pretraining_path + f'/{epoch:02d}_dis.pth')

#             print(f"[D loss: {np.mean(d_loss_list)}] [G loss: {np.mean(g_loss_list)}]")
#         print('Finished Pre Training GAN')
#         print('\n')


#     return generator, discriminator

from models.BigGAN import Generator, Discriminator, G_D
from gan_trainer import train_fns
from data.utils import renormalize_to_standard
import matplotlib.pyplot as plt
import torchvision

def gan_pretraining(classifier, train_loader, z_, y_, fixed_z, fixed_y, args):

    G = Generator(n_classes=args.n_unlabeled_classes, dim_z=args.latent_dim, resolution=args.img_size).to(args.device)
    D = Discriminator(n_classes=args.n_unlabeled_classes, resolution=args.img_size).to(args.device)


    # Load the state_dict into the initialized models if pretrained models are available
    if args.pretrained_gan and os.path.exists(os.path.join(args.pretraining_path, 'G.pth')):
        G.load_state_dict(torch.load(os.path.join(args.pretraining_path, 'G.pth')))
        D.load_state_dict(torch.load(os.path.join(args.pretraining_path, 'D.pth')))

        print("Loaded pre-trained GAN models.")

        return G, D
    
    # if args.pretrained_gan and os.path.exists(args.pretrained_G) and os.path.exists(args.pretrained_D):
    #     G = torch.load(args.pretrained_G).to(args.device)
    #     D = torch.load(args.pretrained_D).to(args.device)
    #     print("Loaded pre-trained GAN models.")

    os.makedirs(args.pretraining_path, exist_ok=True)
    args.img_pretraining_path = os.path.join(args.pretraining_path, 'images')
    os.makedirs(args.img_pretraining_path, exist_ok=True)

    print("Training GAN models from scratch.")

    
    GD = G_D(G, D)

    # GAN training function
    train = train_fns.GAN_training_function(G, D, GD, z_, y_, args.batch_size, args.num_D_steps, num_D_accumulations=1, num_G_accumulations=1)

    # Initialize tracking variables for itertaion wise
    pretrain_itr = 1
    G_losses = []
    D_losses_real = []
    D_losses_fake = []

    # Initialize tracking variables for epoch-wise losses
    epoch_G_losses = []
    epoch_D_losses_real = []
    epoch_D_losses_fake = []

    # Training loop
    for epoch in range(args.n_epochs_gan_pretraining):
        G_loss_epoch = 0
        D_loss_real_epoch = 0
        D_loss_fake_epoch = 0

        num_batches = 0

        for i, (images, targets, idx) in enumerate(tqdm(train_loader)):
            x = images.to(args.device)
            # y = (targets - 5).to(args.device)
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

        print(f"Epoch {epoch + 1}/{args.n_epochs_gan_pretraining}G_loss: {G_loss_epoch_avg:.4f}, D_loss_real: {D_loss_real_epoch_avg:.4f}, D_loss_fake: {D_loss_fake_epoch_avg:.4f}")

        # Save models after each interval
        if(epoch == args.n_epochs_gan_pretraining-1 or ((epoch+1) % args.pretrain_save_interval == 0)):

            # Save only the state_dict of G and D
            torch.save(G.state_dict(), os.path.join(args.models_pretraining_path, 'G.pth'))
            torch.save(D.state_dict(), os.path.join(args.models_pretraining_path, 'D.pth'))


            print(f"GAN model saved at epoch {epoch+1}")

            # Generate sample image
            G.eval()
            with torch.no_grad():
                fixed_Gz = nn.parallel.data_parallel(G, (fixed_z, G.shared(fixed_y)))
            image_filename = os.path.join(args.img_pretraining_path, f'fixed_sample{epoch + 1}.jpg')
            torchvision.utils.save_image(fixed_Gz.float().cpu(), image_filename, nrow=int(fixed_Gz.shape[0] ** 0.5), normalize=True)
            print(f"Sample images saved at {image_filename}.")


    # Save losses in pretraining path
    loss_data = {'G_losses': G_losses, 'D_losses_real': D_losses_real, 'D_losses_fake': D_losses_fake}
    np.save(os.path.join(args.pretraining_path, 'gan_pretraining_iter_losses.npy'), loss_data)
    print("Pretrining Iterwise Loss data saved.")

    # Plot and save the loss curves
    plt.figure()
    plt.plot(G_losses, label="G_loss")
    plt.plot(D_losses_real, label="D_loss_real")
    plt.plot(D_losses_fake, label="D_loss_fake")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("GAN Pretraining Loss (Iter-wise)")
    plt.savefig(os.path.join(args.pretraining_path, 'gan_pretraining_iter_loss_plot.png'))

    # Save epoch-wise loss data
    epoch_loss_data = {'epoch_G_losses': epoch_G_losses, 'epoch_D_losses_real': epoch_D_losses_real, 'epoch_D_losses_fake': epoch_D_losses_fake}
    np.save(os.path.join(args.pretraining_path, 'gan_pretraining_epoch_losses.npy'), epoch_loss_data)
    print("Pretraining Epoch-wise loss data saved.")

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

    print("GAN pretraining Done...")

    
    return G, D
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



# def cls_pretraining(classifier, loader_train, loader_test, learning_rate, n_epochs, results_path):
#     """

#     Args:
#         classifier (TYPE):
#         loader_train (TYPE):
#         loader_test (TYPE):
#         learning_rate (TYPE):
#         n_epochs (TYPE):
#         results_path (TYPE):

#     Returns:
#         Trained classifier on loader
#     """

#     # Pretraining paths
#     cls_pretrained_path = ''.join([results_path, '/cls_pretrained.pth'])

#     device = next(classifier.parameters()).device

#     criterion = nn.CrossEntropyLoss().to(device)
#     optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)

#     os.makedirs(results_path, exist_ok=True)

#     if os.path.isfile(cls_pretrained_path):
#         classifier.load_state_dict(torch.load(cls_pretrained_path))
#         print('loaded existing model')

#     else:
#         print('Starting Training classifier')
#         for epoch in range(n_epochs):  # loop over the dataset multiple times
#             running_loss = 0.0

#             for inputs, labels in loader_train:
#                 inputs, labels = inputs.to(device), labels.to(device)
#                 loss = classifier_train_step(classifier, inputs, optimizer, criterion, labels)
#                 running_loss += loss

#             print(f'Epoch: {epoch} || loss: {running_loss}')
#             if (epoch + 1) % 10 == 0:
#                 print(f'Test accuracy: {100*accuracy(classifier, loader_test):.2f}%')
#         print('Finished Training classifier')
#         print('\n')

#     print('Results:')
#     print(f'Test accuracy: {100*accuracy(classifier, loader_test):.2f}%')
#     torch.save(classifier.state_dict(), cls_pretrained_path)

#     return classifier
def test(model, test_loader, args, tsne=False):
    model.eval()
    preds=np.array([])
    targets=np.array([])
    feats = np.zeros((len(test_loader.dataset), args.n_clusters))
    probs= np.zeros((len(test_loader.dataset), args.n_clusters))
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
        probs[idx, :] = prob.cpu().detach().numpy()
    acc, nmi, ari = cluster_acc(targets.astype(int), preds.astype(int)), nmi_score(targets, preds), ari_score(targets, preds)
    print('Pretrained Test acc {:.4f}, nmi {:.4f}, ari {:.4f}'.format(acc, nmi, ari))
    probs = torch.from_numpy(probs)

    if tsne:
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
        # tsne plot
         # Create t-SNE visualization
        X_embedded = TSNE(n_components=2).fit_transform(feats)  # Use meaningful features for t-SNE

        plt.figure(figsize=(8, 6))
        plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=targets, cmap='viridis')
        plt.title("t-SNE Visualization of Learned Features on Unlabelled CIFAR-10 Subset")
        plt.savefig(args.model_folder+'/tsne.png')
    return acc, nmi, ari, probs

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
    model_dict = torch.load(args.cls_pretraining_path, map_location=args.device)
    model = ResNet(BasicBlock, [2,2,2,2], args.n_unlabeld_classes).to(args.device)
    model.load_state_dict(model_dict['state_dict'], strict=False)
    model.center = Parameter(model_dict['center'])

    # test(model, eval_loader, args)
    # pseudoLabels = getPsedoLabels(model, train_loader, args)

    

    # classifier = ResNet(BasicBlock, [2,2,2,2], args.n_classes).to(args.device)

    # # Define the loss function and optimizer
    # criterion = nn.CrossEntropyLoss().to(args.device)
    # optimizer = optim.Adam(classifier.parameters(), lr=args.lr_cls_pretraining)

    # optimizer = optim.SGD(classifier.parameters(), lr=args.lr_cls_pretraining, momentum=args.momentum, weight_decay=args.weight_decay)

    # # Convert pseudoLabels to a tensor and move it to the device
    # pseudoLabels = torch.from_numpy(pseudoLabels).long().to(args.device)

    # # Training loop
    # for epoch in range(args.n_epochs_cls_pretraining):

    #     loss_record = AverageMeter()

    #     for i, ((images, _), _, _) in enumerate(tqdm(train_loader)):
    #         images = images.to(args.device)
    #         psuedoLabel = pseudoLabels[i*args.batch_size:(i+1)*args.batch_size]

    #         loss = classifier_train_step(classifier, images, optimizer, criterion, psuedoLabel)

    #         loss_record.update(loss, images.size(0))

    #     print(f'Epoch: {epoch} / {args.n_epochs_cls_pretraining}: Avg loss: {loss_record.avg}')

   
        
    return model


def gan_pretraining(generator, discriminator, classifier, loader_train,
                    lr_g, lr_d, latent_dim, n_classes, n_epochs,
                    img_size, results_path):
    """
    Args:
        generator (TYPE)
        discriminator (TYPE)
        classifier (TYPE)
        loader_train (TYPE)
        lr_g (TYPE)
        lr_d (TYPE)
        latent_dim (TYPE)
        n_classes (TYPE)
        n_epochs (TYPE)
        img_size (TYPE)
        results_path (TYPE)

    Returns:
        Pretrained generator and discriminator
    """
    img_pretraining_path = ''.join([results_path, '/images'])
    models_pretraining_path = ''.join([results_path, '/gan_models'])

    g_pretrained = ''.join([results_path, '/generator_pretrained.pth'])
    d_pretrained = ''.join([results_path, '/discriminator_pretrained.pth'])

    device = next(classifier.parameters()).device

    os.makedirs(img_pretraining_path, exist_ok=True)
    os.makedirs(models_pretraining_path, exist_ok=True)

    loaded_gen = False
    loaded_dis = False

    if os.path.isfile(g_pretrained):
        generator.load_state_dict(torch.load(g_pretrained))
        print('loaded existing generator')
        loaded_gen = True

    if os.path.isfile(d_pretrained):
        discriminator.load_state_dict(torch.load(d_pretrained))
        print('loaded existing discriminator')
        loaded_dis = True

    

    if not(loaded_gen and loaded_dis):
        print('Starting Pre Training GAN')
        for epoch in range(n_epochs):
            print(f'Starting epoch {epoch}/{n_epochs}...', end=' ')
            g_loss_list = []
            d_loss_list = []
            for i, ((images, _), _, _) in enumerate(tqdm(loader_train)):

                # real_images = Variable(images).to(device)
                # _, labels = torch.max(classifier(real_images), dim=1)

                real_images = Variable(images).to(device)
                feat = classifier(real_images)
                prob = feat2prob(feat, classifier.center)
                _, labels = prob.max(1)

                generator.train()

                d_loss = discriminator_train_step(discriminator, generator, d_optimizer, criterion_gan,
                                                  real_images, labels, latent_dim, n_classes)
                d_loss_list.append(d_loss)

                g_loss = generator_train_step(discriminator, generator, g_optimizer, criterion_gan,
                                              loader_train.batch_size, latent_dim, n_classes=n_classes)
                g_loss_list.append(g_loss)

            generator.eval()

            latent_space = Variable(torch.randn(n_classes, latent_dim)).to(device)
            gen_labels = Variable(torch.LongTensor(np.arange(n_classes))).to(device)

            gen_imgs = generator(latent_space, gen_labels).view(-1, 3, img_size, img_size)

            if epoch == n_epochs - 1:

                save_image(gen_imgs.data, img_pretraining_path + f'/epoch_{epoch:02d}.png', nrow=n_classes, normalize=True)
                torch.save(generator.state_dict(), models_pretraining_path + f'/{epoch:02d}_gen.pth')
                torch.save(discriminator.state_dict(), models_pretraining_path + f'/{epoch:02d}_dis.pth')

            print(f"[D loss: {np.mean(d_loss_list)}] [G loss: {np.mean(g_loss_list)}]")
        print('Finished Pre Training GAN')
        print('\n')


    return generator, discriminator

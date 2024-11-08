import torch
import torchvision.transforms as transforms
import numpy as np
from data.utils import TransformTwice, TransformKtimes, RandomTranslateWithReflect, TwoStreamBatchSampler
from torch.utils.data import DataLoader, TensorDataset
from GAN_denoise import Generator


# Define constants for CIFAR-10 normalization
mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)


# Initialize and load generator model
z_dim = 128
generator = Generator(z_dim=z_dim).cuda()
generator.load_state_dict(torch.load('/kaggle/input/newgan5to9/pytorch/default/1/generator.pth'))
generator.eval()

print('loaded generator..')

# Define the transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

# Define the transformations with random augmentations for v1 and v2
augment_transform = transforms.Compose([
    # Assuming this class is defined elsewhere as per your setup
    RandomTranslateWithReflect(4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

# Define denormalization transform (inverse of original normalization)
denormalize = transforms.Normalize(
    mean=[-m / s for m, s in zip(mean, std)],
    std=[1 / s for s in std]
)

# Assuming `generator` is your trained generator and `batch_size` is 128
def generate_augmented_samples(generator, batch_size=128, num_batches=200):
    generator.eval()
    all_data = []
    
    with torch.no_grad():
        for _ in range(num_batches):
            # Generate images
            noise = torch.randn(batch_size, z_dim, 1, 1).cuda()
            gen_images = generator(noise).cpu()
            
            # Prepare storage for augmented views
            images, views1, views2 = [], [], []

            for img in gen_images:
                # Denormalize
                denorm_img = denormalize(img)
                
                # Apply augment transformations
                img_v1 = augment_transform(denorm_img)
                img_v2 = augment_transform(denorm_img)

                # Store original and augmented images
                images.append(img)
                views1.append(img_v1)
                views2.append(img_v2)

            # Stack into tensors
            batch_images = torch.stack(images)
            batch_views1 = torch.stack(views1)
            batch_views2 = torch.stack(views2)

            all_data.append(((batch_images, batch_views1, batch_views2), None, None))  # Adjust if labels/indices needed
    
    return all_data

# Load generated data with DataLoader

class GeneratedDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

# Generate data and wrap in a DataLoader
generated_data = generate_augmented_samples(generator)
generated_loader = DataLoader(GeneratedDataset(generated_data), batch_size=1, shuffle=True)

# Example usage
for batch_idx, ((x, x_v1, x_v2), label, idx) in enumerate(generated_loader):
    # x, x_v1, x_v2 will have your generated images and their views
    pass

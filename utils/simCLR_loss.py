import torch
import torch.nn as nn
import numpy as np

class SimCLR_Loss(nn.Module):
    """
    Implements the SimCLR loss function for contrastive learning using cosine similarity and cross-entropy loss.
    
    Parameters:
    - batch_size (int): The number of positive pairs in each batch.
    - temperature (float): A temperature scaling factor for the cosine similarities, controlling the concentration of the distribution.
    """
    def __init__(self, batch_size, temperature):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature

        # Mask to filter out positive pairs (correlated samples)
        self.mask = self.mask_correlated_samples(batch_size)
         # Cross-entropy loss function to compute the contrastive loss
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
         # Cosine similarity function for pairwise similarity calculation
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size):
        """
        Creates a mask to exclude positive pairs from the set of negative samples.
        
        For a batch of size N, this mask is a (2N, 2N) boolean matrix where positive pairs 
        (corresponding augmented views) are set to 0, and all others to 1. This prevents the 
        positive pairs from being treated as negatives during loss calculation.
        
        Returns:
        - mask (torch.Tensor): A boolean mask of shape (2N, 2N).
        """
        N = 2 * batch_size # 2N total samples due to positive pairs
        mask = torch.ones((N, N), dtype=bool) # Start with all negatives ( mask = 1 )
        mask = mask.fill_diagonal_(0)  # Set diagonal to 0 to exclude self-similarities

        # Mark positive pairs (i, batch_size + i) and (batch_size + i, i) as 0
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        """
        Computes the SimCLR loss for a batch of embeddings.
        
        Parameters:
        - z_i (torch.Tensor): Embeddings from the first set of augmented views. [ batch_size, n_dim]
        - z_j (torch.Tensor): Embeddings from the second set of augmented views. [ batch_size, n_dim]
        
        Returns:
        - loss (torch.Tensor): Computed contrastive loss.
        """

        N = 2 * self.batch_size # Total number of samples (pairs of augmented views)
        
        # Concatenate embeddings from both views to form a single batch : [2*batch_size, n_dim]
        z = torch.cat((z_i, z_j), dim=0) 

        # Compute pairwise cosine similarity for all samples, scaled by temperature: [batch_size, batch_size]
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature 

         # Extract the positive similarities (diagonals) for both directions: 
        sim_i_j = torch.diag(sim, self.batch_size) # Positive Pair Similarity between z_i and z_j : [batch_size]
        sim_j_i = torch.diag(sim, -self.batch_size) # Positive Pair Similarity between z_j and z_i: [batch_size]
        
        # Combine positive similarities (z_i & z_j) into a single tensor (N, 1) for loss calculation: [2*batch_size, 1]
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        # Extract negative samples using the mask and reshape to (N, -1) for loss calculation: [2*batch_size, n_dim - 2]
        negative_samples = sim[self.mask].reshape(N, -1)
        
        #SIMCLR
        # Create labels: All positives are labeled 0, cross-entropy will treat them as correct class: [2*BatchSize]
        labels = torch.from_numpy(np.array([0]*N)).reshape(-1).to(positive_samples.device).long() #.float()
        
        # Concatenate positive and negative samples for cross-entropy computation: [2*batch_size, n_dim - 1]
        logits = torch.cat((positive_samples, negative_samples), dim=1)

        # Compute the contrastive loss using cross-entropy. The first one is the postive one. The logits should predict that as well ( Cross Entorpy)
        loss = self.criterion(logits, labels)
        loss /= N  # Normalize by the batch size
        
        return loss
    

class ClusterBasedContrastiveLoss(nn.Module):
    def __init__(self, hard_pos_k, n_cluster, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        self.hard_pos_k = hard_pos_k
        self.num_clusters = n_cluster
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def forward(self, prob, z_i, z_j):
        batch_size = prob.size(0)
        z = torch.cat((z_i, z_j), dim=0)  # Combine features [2*batch_size, dim]

        # Cluster-wise top-k selection
        topk_indices = torch.topk(prob, self.hard_pos_k, dim=0).indices  # [k, num_clusters]
        print(topk_indices.shape)
        print(topk_indices)

        loss = 0.0
        total_clusters = 0

        for cluster_id in range(self.num_clusters):
            pos_indices = topk_indices[:, cluster_id]  # Top-k indices for cluster c
            pos = z[pos_indices]  # Positives from cluster c
            pos_aug = z[pos_indices + batch_size]  # Augmented positives

            # Combine for contrastive pairs
            pos_pairs = torch.cat([pos, pos_aug], dim=0)  # [2k, dim]


            # Negatives from all other clusters (include both z_i and z_j)
            neg_indices = torch.cat([
                torch.cat([topk_indices[:, c], topk_indices[:, c] + batch_size])
                for c in range(self.num_clusters) if c != cluster_id
            ])
            neg = z[neg_indices]  # [neg_count, dim]

            # Compute similarities
            pos_sim = self.similarity_f(pos_pairs.unsqueeze(1), pos.unsqueeze(0)) / self.temperature
            neg_sim = self.similarity_f(pos_pairs.unsqueeze(1), neg.unsqueeze(0)) / self.temperature

            # Contrastive loss
            pos_loss = -torch.log(torch.exp(pos_sim).sum(dim=1))  # Positive pairs closer
            neg_loss = torch.log(torch.exp(neg_sim).sum(dim=1))   # Negatives pushed away

            cluster_loss = pos_loss + neg_loss
            loss += cluster_loss.mean()  # Average over cluster
            total_clusters += 1

        return loss / total_clusters if total_clusters > 0 else torch.tensor(0.0, device=z.device)


# class ClusterBasedContrastiveLoss(nn.Module):
#     def __init__(self, hard_pos_k, n_cluster, temperature=0.5):
#         super().__init__()
#         self.temperature = temperature
#         self.hard_pos_k = hard_pos_k
#         self.num_clusters = n_cluster
#         self.similarity_f = nn.CosineSimilarity(dim=2)

#     def forward(self, prob, prob_bar, z_i, z_j):
#         batch_size = prob.size(0)
#         z = torch.cat((z_i, z_j), dim=0)  # Combine embeddings
#         prob = torch.cat((prob, prob_bar), dim=0)  # Combine probabilities

#         # Step 1: Assign samples to clusters
#         cluster_assignments = prob.argmax(dim=1)  # Cluster assignment
#         clusters = {c: (cluster_assignments == c).nonzero(as_tuple=True)[0] for c in range(self.num_clusters)}

#         loss = 0.0
#         total_clusters = 0

#         # Step 2: Process each cluster
#         for cluster_id, indices in clusters.items():
#             if len(indices) < self.hard_pos_k:
#                 continue  # Skip if too few samples in the cluster

#             # Select top-k hard positives (highest prob in the cluster)
#             cluster_prob = prob[indices]  # [cluster_size, num_clusters]
#             topk_indices = cluster_prob[:, cluster_id].topk(self.hard_pos_k).indices
#             hard_pos_indices = indices[topk_indices]

#             # Create positives and negatives
#             pos = z[hard_pos_indices]  # [k, embedding_dim]
#             pos_aug = z[hard_pos_indices + batch_size]  # Augmented views

#             # Combine anchor and positive
#             pos_pairs = torch.cat([pos, pos_aug], dim=0)  # [2k, embedding_dim]

#             # Negative samples come from all other clusters
#             neg_indices = torch.cat([clusters[c] for c in range(self.num_clusters) if c != cluster_id])
#             neg = z[neg_indices]  # [neg_count, embedding_dim]

#             # Compute similarities
#             pos_sim = self.similarity_f(pos_pairs.unsqueeze(1), pos.unsqueeze(0)) / self.temperature
#             neg_sim = self.similarity_f(pos_pairs.unsqueeze(1), neg.unsqueeze(0)) / self.temperature

#             # Contrastive loss for the cluster
#             pos_loss = -torch.log(torch.exp(pos_sim).sum(dim=1))
#             neg_loss = torch.log(torch.exp(neg_sim).sum(dim=1))
#             cluster_loss = pos_loss + neg_loss

#             loss += cluster_loss.mean()  # Average over cluster
#             total_clusters += 1

#         return loss / total_clusters if total_clusters > 0 else torch.tensor(0.0, device=z.device)

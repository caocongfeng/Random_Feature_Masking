import torch
import copy
import random
import pdb
import scipy.sparse as sp
import numpy as np

def main():
    pass

def aug_random_mask(input_feature, drop_percent):
    """
    Randomly masks a fixed percentage of features for each node.
    
    Args:
        input_feature (torch.Tensor): A 3D tensor of shape [batch, node, feature].
        drop_percent (float): The percentage of features to be masked per node (0 to 1).
    
    Returns:
        torch.Tensor: The masked feature tensor.
    """
    batch_size, num_nodes, num_features = input_feature.shape
    
    # Calculate the number of features to mask per node
    num_masked_features = int(num_features * drop_percent)
    
    # Generate mask indices for each node
    masked_feature = input_feature.clone()
    for b in range(batch_size):
        for n in range(num_nodes):
            mask_indices = torch.randperm(num_features, device=input_feature.device)[:num_masked_features]
            masked_feature[b, n, mask_indices] = 0
    
    return masked_feature

# def aug_random_mask(input_feature, drop_percent):
#     """
#     Randomly masks individual features across all nodes with the given drop rate.
    
#     Args:
#         input_feature (torch.Tensor): A 3D tensor of shape [batch, node, feature].
#         drop_percent (float): The percentage of features to be masked (0 to 1).
    
#     Returns:
#         torch.Tensor: The masked feature tensor.
#     """
#     batch_size, num_nodes, num_features = input_feature.shape
    
#     # Create a mask for each feature independently
#     mask = torch.rand((batch_size, num_nodes, num_features), device=input_feature.device) < drop_percent
    
#     # Apply the mask (set masked elements to zero)
#     masked_feature = input_feature.clone()
#     masked_feature[mask] = 0
    
#     return masked_feature

# def aug_random_mask(input_feature, drop_percent):
#     """
#     Randomly masks features for each node across the batch with the given drop rate.
    
#     Args:
#         input_feature (torch.Tensor): A 3D tensor of shape [batch, node, feature].
#         drop_percent (float): The percentage of features to be masked (0 to 1).
    
#     Returns:
#         torch.Tensor: The masked feature tensor.
#     """
#     batch_size, num_nodes, num_features = input_feature.shape
    
#     # Create a mask for features per node
#     mask = torch.rand((batch_size, num_nodes), device=input_feature.device) < drop_percent
#     mask = mask.unsqueeze(-1).expand(-1, -1, num_features)  # Expand to feature dimension
    
#     # Apply the mask (set masked elements to zero)
#     masked_feature = input_feature.clone()
#     masked_feature[mask] = 0
    
#     return masked_feature

# def aug_random_mask(input_feature, drop_percent):
#     """
#     Randomly masks node features across all nodes in the batch.
    
#     Args:
#         input_feature (torch.Tensor): A 3D tensor of shape [batch, node, feature].
#         drop_percent (float): The percentage of features to be masked (0 to 1).
    
#     Returns:
#         torch.Tensor: The masked feature tensor.
#     """
#     batch_size, num_nodes, num_features = input_feature.shape
    
#     # Create a mask for each node across all features
#     mask = torch.rand((batch_size, num_nodes), device=input_feature.device) < drop_percent
    
#     # Expand the mask to match feature dimensions
#     mask = mask.unsqueeze(-1).expand(-1, -1, num_features)
    
#     # Apply the mask (set masked elements to zero)
#     masked_feature = input_feature.clone()
#     masked_feature[mask] = 0
    
#     return masked_feature

# def aug_random_mask(input_feature, drop_percent):
#     """
#     Randomly masks node features across all nodes in the batch.
    
#     Args:
#         input_feature (torch.Tensor): A 3D tensor of shape [batch, node, feature].
#         drop_percent (float): The percentage of features to be masked (0 to 1).
    
#     Returns:
#         torch.Tensor: The masked feature tensor.
#     """
#     batch_size, num_nodes, num_features = input_feature.shape
    
#     # Create a mask with the same shape as input_feature
#     mask = torch.empty((batch_size, num_nodes, num_features), device=input_feature.device).uniform_(0, 1) < drop_percent
    
#     # Apply the mask (set masked elements to zero)
#     masked_feature = input_feature.clone()
#     masked_feature[mask] = 0
    
#     return masked_feature

# def aug_random_mask(input_feature, drop_percent):
#     print(input_feature.size())
#     batch,node,feature=input_feature.size()
#     drop_mask = (
#         # !1 change mask
#         torch.empty((node,feature), dtype=torch.float32, device=input_feature.device).uniform_(0, 1)
#         < drop_percent
#     )
#     print(drop_mask)
#     input_feature = input_feature.clone()
#     # !1 change mask
#     input_feature[:,drop_mask] = 0

#     return input_feature
# def aug_random_mask(input_feature, drop_percent):
#     drop_mask = torch.empty(
#         (input_feature.size(1),),
#         dtype=torch.float32,
#         device=input_feature.device).uniform_(0, 1) < drop_percent
#     input_feature = input_feature.clone()
#     input_feature[:, drop_mask] = 0

#     return input_feature

# def aug_random_mask(input_feature, drop_percent=0.2):
#     """
#     Randomly masks features of each node independently across all nodes.
    
#     Args:
#         input_feature (torch.Tensor): A tensor of shape [batch_size, node_num, feature_dim].
#         drop_percent (float): The percentage of features to mask per node.
    
#     Returns:
#         torch.Tensor: Augmented tensor with randomly masked features.
#     """
#     batch_size, node_num, feature_dim = input_feature.shape
#     aug_feature = copy.deepcopy(input_feature)

#     # Determine number of features to drop per node
#     mask_num = int(feature_dim * drop_percent)
    
#     for i in range(batch_size):  # Iterate over each batch
#         for j in range(node_num):  # Iterate over each node
#             feature_idx = list(range(feature_dim))
#             mask_idx = random.sample(feature_idx, mask_num)  # Select features to mask
            
#             aug_feature[i, j, mask_idx] = 0  # Zero out selected features

#     return aug_feature


# def aug_random_mask(input_feature, drop_percent=0.2):
    
#     node_num = input_feature.shape[1]
#     mask_num = int(node_num * drop_percent)
#     node_idx = [i for i in range(node_num)]
#     mask_idx = random.sample(node_idx, mask_num)
#     aug_feature = copy.deepcopy(input_feature)
#     zeros = torch.zeros_like(aug_feature[0][0])
#     for j in mask_idx:
#         aug_feature[0][j] = zeros
#     return aug_feature

# def aug_random_mask(input_feature, drop_percent):
#     drop_mask = (
#         # !1 change mask
#         torch.empty((input_feature.size()), dtype=torch.float32, device=input_feature.device).uniform_(0, 1)
#         < drop_percent
#     )
#     input_feature = input_feature.clone()
#     # !1 change mask
#     input_feature[drop_mask] = 0

#     return input_feature
# def aug_random_mask(input_feature, drop_percent):
#     """
#     Applies random feature masking independently for each batch.

#     Args:
#         input_feature (torch.Tensor): A tensor of shape [batch_size, node_num, feature_dim].
#         drop_percent (float): Probability of masking each feature value.

#     Returns:
#         torch.Tensor: Augmented tensor with randomly masked feature values.
#     """
#     # Create a mask independently for each batch
#     drop_mask = torch.rand_like(input_feature, dtype=torch.float32, device=input_feature.device) < drop_percent

#     # Clone the input tensor to avoid modifying the original
#     aug_feature = input_feature.clone()

#     # Apply the mask
#     aug_feature[drop_mask] = 0

#     return aug_feature
# import torch

# def aug_random_mask(input_feature, drop_percent):
#     """
#     Applies random feature masking independently for each batch.

#     Args:
#         input_feature (torch.Tensor): A tensor of shape [batch_size, node_num, feature_dim].
#         drop_percent (float): Probability of masking each feature value.

#     Returns:
#         torch.Tensor: Augmented tensor with randomly masked feature values.
#     """
#     batch_size, node_num, feature_dim = input_feature.shape

#     # Create a mask independently for each batch
#     drop_mask = torch.rand((batch_size, node_num, feature_dim), device=input_feature.device) < drop_percent

#     # Clone the input tensor to avoid modifying the original
#     aug_feature = input_feature.clone()

#     # Apply the mask independently per batch
#     aug_feature[drop_mask] = 0

#     return aug_feature


# def aug_random_mask(input_feature, drop_percent):
#     """
#     Randomly masks node features across all nodes for each batch.

#     Args:
#         input_feature (torch.Tensor): A tensor of shape [batch_size, node_num, feature_dim].
#         drop_percent (float): Probability of masking each feature across all nodes in a batch.

#     Returns:
#         torch.Tensor: Augmented tensor with randomly masked feature values.
#     """
#     batch_size, node_num, feature_dim = input_feature.shape

#     # Create a mask of shape (batch_size, 1, feature_dim) to apply the same masking across all nodes per batch & feature
#     drop_mask = torch.rand((batch_size, 1, feature_dim), device=input_feature.device) < drop_percent

#     # Expand mask to match node dimension, ensuring same masking across all nodes for each batch & feature
#     drop_mask = drop_mask.expand(batch_size, node_num, feature_dim)

#     # Clone the input tensor to avoid modifying the original
#     aug_feature = input_feature.clone()

#     # Apply the mask
#     aug_feature[drop_mask] = 0

#     return aug_feature


# def drop_feature(x, drop_prob):
#     drop_mask = (
#         # !1 change mask
#         th.empty((x.size()), dtype=th.float32, device=x.device).uniform_(0, 1)
#         < drop_prob
#     )
#     x = x.clone()
#     # !1 change mask
#     x[drop_mask] = 0

#     return x


# def aug_random_mask(input_feature, drop_percent):
#     drop_mask = (
#         # !1 change mask
#         torch.empty((input_feature.size()), dtype=torch.float32, device=input_feature.device).uniform_(0, 1)
#         < drop_percent
#     )
#     input_feature = input_feature.clone()
#     # !1 change mask
#     input_feature[drop_mask] = 0

#     return input_feature

def aug_random_edge(input_adj, drop_percent=0.2):

    percent = drop_percent / 2
    row_idx, col_idx = input_adj.nonzero()

    index_list = []
    for i in range(len(row_idx)):
        index_list.append((row_idx[i], col_idx[i]))

    single_index_list = []
    for i in list(index_list):
        single_index_list.append(i)
        index_list.remove((i[1], i[0]))
    
    
    edge_num = int(len(row_idx) / 2)      # 9228 / 2
    add_drop_num = int(edge_num * percent / 2) 
    aug_adj = copy.deepcopy(input_adj.todense().tolist())

    edge_idx = [i for i in range(edge_num)]
    drop_idx = random.sample(edge_idx, add_drop_num)

    
    for i in drop_idx:
        aug_adj[single_index_list[i][0]][single_index_list[i][1]] = 0
        aug_adj[single_index_list[i][1]][single_index_list[i][0]] = 0
    
    '''
    above finish drop edges
    '''
    node_num = input_adj.shape[0]
    l = [(i, j) for i in range(node_num) for j in range(i)]
    add_list = random.sample(l, add_drop_num)

    for i in add_list:
        
        aug_adj[i[0]][i[1]] = 1
        aug_adj[i[1]][i[0]] = 1
    
    aug_adj = np.matrix(aug_adj)
    aug_adj = sp.csr_matrix(aug_adj)
    return aug_adj


def aug_drop_node(input_fea, input_adj, drop_percent=0.2):

    input_adj = torch.tensor(input_adj.todense().tolist())
    input_fea = input_fea.squeeze(0)

    node_num = input_fea.shape[0]
    drop_num = int(node_num * drop_percent)    # number of drop nodes
    all_node_list = [i for i in range(node_num)]

    drop_node_list = sorted(random.sample(all_node_list, drop_num))

    aug_input_fea = delete_row_col(input_fea, drop_node_list, only_row=True)
    aug_input_adj = delete_row_col(input_adj, drop_node_list)

    aug_input_fea = aug_input_fea.unsqueeze(0)
    aug_input_adj = sp.csr_matrix(np.matrix(aug_input_adj))

    return aug_input_fea, aug_input_adj


def aug_subgraph(input_fea, input_adj, drop_percent=0.2):
    
    input_adj = torch.tensor(input_adj.todense().tolist())
    input_fea = input_fea.squeeze(0)
    node_num = input_fea.shape[0]

    all_node_list = [i for i in range(node_num)]
    s_node_num = int(node_num * (1 - drop_percent))
    center_node_id = random.randint(0, node_num - 1)
    sub_node_id_list = [center_node_id]
    all_neighbor_list = []

    for i in range(s_node_num - 1):
        
        all_neighbor_list += torch.nonzero(input_adj[sub_node_id_list[i]], as_tuple=False).squeeze(1).tolist()
        
        all_neighbor_list = list(set(all_neighbor_list))
        new_neighbor_list = [n for n in all_neighbor_list if not n in sub_node_id_list]
        if len(new_neighbor_list) != 0:
            new_node = random.sample(new_neighbor_list, 1)[0]
            sub_node_id_list.append(new_node)
        else:
            break

    
    drop_node_list = sorted([i for i in all_node_list if not i in sub_node_id_list])

    aug_input_fea = delete_row_col(input_fea, drop_node_list, only_row=True)
    aug_input_adj = delete_row_col(input_adj, drop_node_list)

    aug_input_fea = aug_input_fea.unsqueeze(0)
    aug_input_adj = sp.csr_matrix(np.matrix(aug_input_adj))

    return aug_input_fea, aug_input_adj





def delete_row_col(input_matrix, drop_list, only_row=False):

    remain_list = [i for i in range(input_matrix.shape[0]) if i not in drop_list]
    out = input_matrix[remain_list, :]
    if only_row:
        return out
    out = out[:, remain_list]

    return out



    



    

     

    







if __name__ == "__main__":
    main()
    

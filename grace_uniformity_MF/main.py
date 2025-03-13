import argparse
import warnings

import numpy as np
import torch as th
import torch.nn as nn
from aug import aug
from dataset import load
from eval import label_classification
from model import Grace

from sklearn import manifold
from scipy.stats import gaussian_kde
import os


import random
import time

warnings.filterwarnings("ignore")


def count_parameters(model):
    return sum(
        [np.prod(p.size()) for p in model.parameters() if p.requires_grad]
    )


parser = argparse.ArgumentParser()
parser.add_argument("--dataname", type=str, default="cora")
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--split", type=str, default="random")

parser.add_argument(
    "--epochs", type=int, default=500, help="Number of training periods."
)
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
parser.add_argument("--wd", type=float, default=1e-5, help="Weight decay.")
parser.add_argument("--temp", type=float, default=1.0, help="Temperature.")

parser.add_argument("--act_fn", type=str, default="relu")

parser.add_argument(
    "--hid_dim", type=int, default=256, help="Hidden layer dim."
)
parser.add_argument(
    "--out_dim", type=int, default=256, help="Output layer dim."
)

parser.add_argument(
    "--num_layers", type=int, default=2, help="Number of GNN layers."
)
parser.add_argument(
    "--der1",
    type=float,
    default=0.2,
    help="Drop edge ratio of the 1st augmentation.",
)
parser.add_argument(
    "--der2",
    type=float,
    default=0.2,
    help="Drop edge ratio of the 2nd augmentation.",
)
parser.add_argument(
    "--dfr1",
    type=float,
    default=0.2,
    help="Drop feature ratio of the 1st augmentation.",
)
parser.add_argument(
    "--dfr2",
    type=float,
    default=0.2,
    help="Drop feature ratio of the 2nd augmentation.",
)

parser.add_argument("--name", type=str, default="result")

args = parser.parse_args()

if args.gpu != -1 and th.cuda.is_available():
    args.device = "cuda:{}".format(args.gpu)
else:
    args.device = "cpu"

if __name__ == "__main__":
    # Step 1: Load hyperparameters =================================================================== #


    lr = args.lr
    hid_dim = args.hid_dim
    out_dim = args.out_dim

    num_layers = args.num_layers
    act_fn = ({"relu": nn.ReLU(), "prelu": nn.PReLU()})[args.act_fn]

    drop_edge_rate_1 = args.der1
    drop_edge_rate_2 = args.der2
    drop_feature_rate_1 = args.dfr1
    drop_feature_rate_2 = args.dfr2

    temp = args.temp
    epochs = args.epochs
    wd = args.wd

    # Step 2: Prepare data =================================================================== #
    graph, feat, labels, train_mask, test_mask = load(args.dataname)
    in_dim = feat.shape[1]

    # Step 3: Create model =================================================================== #
    model = Grace(in_dim, hid_dim, out_dim, num_layers, act_fn, temp)
    model = model.to(args.device)
    print(f"# params: {count_parameters(model)}")

    optimizer = th.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    # Step 4: Training =======================================================================
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        graph1, feat1 = aug(graph, feat, drop_feature_rate_1, drop_edge_rate_1)
        graph2, feat2 = aug(graph, feat, drop_feature_rate_2, drop_edge_rate_2)

        graph1 = graph1.to(args.device)
        graph2 = graph2.to(args.device)

        feat1 = feat1.to(args.device)
        feat2 = feat2.to(args.device)

        loss = model(graph1, graph2, feat1, feat2)
        loss.backward()
        optimizer.step()

        print(f"Epoch={epoch:03d}, loss={loss.item():.4f}")

    # Step 5: Linear evaluation ============================================================== #
    print("=== Final ===")

    graph = graph.add_self_loop()
    graph = graph.to(args.device)
    feat = feat.to(args.device)
    embeds = model.get_embedding(graph, feat)
    X=embeds
    
    X = X.cpu().detach().numpy()
    tsne = manifold.TSNE(n_components=2)
    X = tsne.fit_transform(X)
    X_norm = X / np.linalg.norm(X, axis=1)[:, None]

    m1 = X_norm[:, 0]
    m2 = X_norm[:, 1]
    xmin = -1.3
    xmax = 1.3
    ymin = -1.3
    ymax = 1.3
    Xx, Y = np.mgrid[xmin:xmax:1000j, ymin:ymax:1000j]
    positions = np.vstack([Xx.ravel(), Y.ravel()])
    values = np.vstack([m1, m2])
    kernel = gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, Xx.shape)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 13))
    fig, ax = plt.subplots()
    ax.imshow(np.rot90(Z), cmap='GnBu', alpha=1, vmin=0.18,
                extent=[-1.1, 1.1, -1.1, 1.1])
    # plt.rcParams['font.family'] = 'Times New Roman'
    plt.xlim((-1.1, 1.1))
    plt.ylim((-1.1, 1.1))
    plt.xticks([-0.8, 0, 0.8], ['-1', '0', '1'], size=20)
    plt.yticks([-0.8, 0, 0.8], ['-1', '0', '1'], size=20)
    plt.ylabel('Features', size=20)
    plt.title('Feature Distribution', size=25)
    
    formatted_time = time.strftime("%y_%m_%d_%H_%M_%S")

    random_feature_masking='False'

    path=args.name+'/'+args.dataname+'/'+str(args.dataname)+'_'+str(args.der1)+"_"+str(args.dfr2)+'_'+str(temp)+"_"+random_feature_masking+'_'+formatted_time

    th.save(embeds,path+'_embeds.pt')

    filename = path+'_distribution.pdf'
    plt.savefig(filename)

    """Evaluation Embeddings  """
    label_classification(
        embeds, labels, train_mask, test_mask, split=args.split
    )

import os.path

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from sklearn.metrics import roc_auc_score, average_precision_score
from model import *  # 确保导入正确的模型
from data_processing import *
from data_processing import feature_extract_extraction
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics  import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve, auc
from add_noise import add_random_edges


# 路径设置
source_path = "E:\GAT\data\HDST\mouse_olfactory_bulb//"
add_false_ratio_path = source_path + "//0%//"
adj_path = source_path + "/original_adj_matrix.csv"
coord_path = source_path + "/CN13_D2_X_Y.tsv"
platform = "HDST"
image_path = source_path + "/CN13_D2_HE.png"
save_path = source_path + "/patch/"
# scale_path = source_path + "scalefactors_json.json"
scale_path = None
# 检查并加载邻接矩阵
if os.path.exists(adj_path):
    adj = pd.read_csv(adj_path, index_col=0)
else:
    adj = genarate_adj_matrix(coord_path, platform=platform)  # 修复函数名拼写错误
    adj = pd.DataFrame(adj)
    adj.to_csv(source_path + "/original_adj_matrix.csv")
adj_matrix = adj
adj = csr_matrix(adj)
adj_orig = adj.copy()
adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
adj_orig.eliminate_zeros()
# 切分数据集
adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
adj_train_noise = add_random_edges(adj_train,proportion=0)
# print(adj_train)
# print(type(adj_train))
print(adj_train.shape[0])
adj = adj_train_noise


# 一些预处理操作
adj_norm = preprocess_graph(adj)
num_nodes = adj.shape[0]

# 创建模型
pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()

norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

adj_label = adj_train + sp.eye(adj_train.shape[0])
adj_label = sparse_to_tuple(adj_label)

adj_norm = torch.sparse.FloatTensor(torch.LongTensor(adj_norm[0].T),
                                    torch.FloatTensor(adj_norm[1]),
                                    torch.Size(adj_norm[2]))

adj_label = torch.sparse.FloatTensor(torch.LongTensor(adj_label[0].T),
                                     torch.FloatTensor(adj_label[1]),
                                     torch.Size(adj_label[2]))

weight_mask = adj_label.to_dense().view(-1) == 1
weight_tensor = torch.ones(weight_mask.size(0))
weight_tensor[weight_mask] = pos_weight

def get_scores(edges_pos, edges_neg, adj_rec):
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(adj_rec[e[0],e[1]].item())
        pos.append(adj_orig[e[0],e[1]])
    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(adj_rec[e[0],e[1]].data)
        neg.append(adj_orig[e[0],e[1]])

    preds_all = np.hstack([preds,preds_neg])
    labels_all = np.hstack([np.ones(len(preds)),np.zeros(len(preds_neg))])

    roc_score = roc_auc_score(labels_all,preds_all)
    ap_score = average_precision_score(labels_all,preds_all)
    acc_score = accuracy_score(labels_all, np.round(preds_all))
    F1_score = f1_score(labels_all, np.round(preds_all))
    precision, recall, _ = precision_recall_curve(labels_all, preds_all)
    auprc = auc(recall, precision)

    return roc_score,ap_score,acc_score,F1_score,auprc

# 计算精度
def get_acc(adj_rec, adj_label):
    labels_all = adj_label.to_dense().view(-1).long()
    preds_all = (adj_rec > 0.5).view(-1).long()
    accuracy = (preds_all == labels_all).sum().float() / labels_all.size(0)
    return accuracy


# 训练过程
train_acc_all = []
val_auc_all = []
val_ap_all = []
val_acc_all = []
val_f1_all = []
val_prc_all = []
def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output,reconstruct_feature,reconstruct_image_feature = model(features,image_features, adj)
    loss_train = norm * F.binary_cross_entropy(output.view(-1), adj_label.to_dense().view(-1), weight=weight_tensor)
    loss2 = nn.MSELoss()(features, reconstruct_feature)
    loss3 = nn.MSELoss()(image_features, reconstruct_image_feature)
    loss_train = 0.6 * loss_train + 0.2 * loss2 + 0.2 * loss3
    acc_train = get_acc(output, adj_label)
    loss_train.backward()
    train_acc_all.append((epoch,acc_train.item()))
    # 梯度剪裁，避免梯度过大导致不稳定
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    model.eval()
    output,reconstruct_feature ,reconstruct_image_feature = model(features, image_features,adj)
    val_roc, val_ap,val_acc,val_f1,val_prc  = get_scores(val_edges, val_edges_false, output)
    val_auc_all.append((epoch,val_roc.item()))
    val_ap_all.append((epoch,val_ap.item()))
    val_acc_all.append((epoch, val_acc.item()))
    val_f1_all.append((epoch, val_f1.item()))
    val_prc_all.append((epoch,val_prc.item()))
    print('Epoch: {:04d}'.format(epoch + 1),
          'train_loss: {:.4f}'.format(loss_train.data.item()),
          'train_acc: {:.4f}'.format(acc_train.data.item()),
          'val_roc: {:.4f}'.format(val_roc),
          'val_ap: {:.4f}'.format(val_ap),
          'val_acc: {:.4f}'.format(val_acc),
          'val_f1: {:.4f}'.format(val_f1),
          'val_auprc: {:.4f}'.format(val_prc),
          'time: {:.4f}s'.format(time.time() - t))
    return loss_train.data.item()  # 保存损失


# 测试集上的结果
def compute_test():
    model.eval()
    A_pred , reconstruct_feature ,reconstruct_image_feature = model(features, image_features ,adj)
    test_roc, test_ap,test_acc,test_f1,test_prc  = get_scores(test_edges, test_edges_false, A_pred)
    print("Test set results:",
          "test_roc= {:.4f}".format(test_roc),
          "test_ap= {:.4f}".format(test_ap),
          "test_acc= {:.4f}".format(test_acc),
          "test_f1= {:.4f}".format(test_f1),
          "test_prc= {:.4f}".format(test_prc))
    return A_pred ,test_roc, test_ap,test_acc,test_f1,test_prc


# 主函数
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--hidden', type=int, default=256, help='hidden size')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--nheads', type=int, default=8, help='Number of head attentions')
    parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
    parser.add_argument('--patience', type=int, default=100, help='Patience')
    parser.add_argument('--seed', type=int, default=17, help='Seed number')
    args = parser.parse_args()

    # 随机数种子设置
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    adj = adj_norm
    adj = adj.to_dense()

    # 加载特征数据
    features = pd.read_csv(source_path + "CN13_D2_count.tsv", sep="\t",index_col=0)
    print(features)
    features = torch.tensor(features.to_numpy()).to(torch.float32)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(add_false_ratio_path):
        os.makedirs(add_false_ratio_path)
    # 加载图像特征数据
    if os.path.exists(source_path+"//image_matrix.csv"):
        image_matrix = pd.read_csv(source_path+"/image_matrix.csv",sep=",")
        image_matrix = image_matrix.loc[:,~image_matrix.columns.str.contains("Unname")]
        image_matrix = np.array(image_matrix)
        indices = np.argsort(image_matrix,axis=1)[:,-10:]
        result = np.zeros_like(image_matrix)
        rows = np.arange(image_matrix.shape[0])[:,None]
        result[rows,indices] = 1
        image_matrix = torch.tensor(result).to(torch.float32)
        image_features = image_matrix
        print("iamge_matrix_file already exis!!!!!!")
    else:
        image_matrix = feature_extract_extraction(save_path,coord_path,platform = platform,image_path=image_path,scale_path=scale_path)
        image_features = torch.tensor(image_matrix.to_numpy()).to(torch.float32)

    # 标准化操作：将数据归一化为均值为0，标准差为1
    min_vals = torch.min(features, dim=0).values
    max_vals = torch.max(features, dim=0).values
    features = (features - min_vals) / (max_vals - min_vals)
    # print(features)

    model = GAT(input_size=features.shape[1], input_feature_size = image_features.shape[1],hidden_size=args.hidden, output_size=128,
                dropout=args.dropout, nheads=args.nheads, alpha=args.alpha)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.8, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.05, patience=10)

    t_total = time.time()
    loss_values = []
    bad_counter = 0
    best_loss = float('inf')
    best_epoch = 0

    for epoch in range(args.epochs):
        loss = train(epoch)
        loss_values.append(loss)

        # if loss < best_loss:
        #     best_loss = loss
        #     best_epoch = epoch
        #     bad_counter = 0
        # else:
        #     bad_counter += 1
        #
        # if bad_counter == args.patience:
        #     break


    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    A_pred , test_roc,test_ap ,test_acc ,test_f1,test_prc = compute_test()


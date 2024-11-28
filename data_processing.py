from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from tensorflow.keras.preprocessing import image
import cv2
import pandas as pd
import numpy as np
import os
from scipy import spatial
from tqdm import tqdm
from PIL import Image
from pathlib import Path
import json
from sklearn.decomposition import PCA

def split_and_save_patch(image_path, patch_size, coordinates, save_dir):
    """
    根据坐标位置，以及patch的大小切分数据
    Args:
        image_path: 图像特征的路径
        patch_size: 要切分每个patch的尺寸
        coordinates: 坐标
        save_dir: 切分后每个patch的保存路径

    Returns:

    """

    image = cv2.imread(image_path)
    for i, coord in enumerate(coordinates):
        x, y = coord
        patch = image[y:y+patch_size[1], x:x+patch_size[0]]
        save_path = os.path.join(save_dir, f'patch_{i}.png')
        cv2.imwrite(save_path, patch)

def RNet50(img_path):
    """
    使用Tensorflow中训练好的Resnet50提取每个patch的特征
    Args:
        img_path:每个小patch的路径

    Returns:提取每个patch的特征

    """
    model = ResNet50(weights='imagenet', include_top=False)
    img_path = img_path
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    feature = model.predict(x)
    return feature
def read_json(json_file):
    """
    读取10X数据的.json文件，确定图像的缩放比例
    Args:
        json_file: .json的文件名

    Returns:读取后的数据

    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data
def image_crop(image_path , save_path , coord ,scale ,platform,crop_size=50,target_size=224,verbose=False):
    """
    根据不同平台切分并提取patch特征
    Args:
        image_path: 图像路径
        save_path: 保存patch的路径
        coord: 坐标信息
        scale: 缩放比例
        platform: 平台
        crop_size: 切分大小
        target_size: 目标大熊啊
        verbose:

    Returns:

    """
    if platform == "10X":
        image = cv2.imread(image_path)
    else:
        image = image_path
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = (image * 255).astype(np.uint8)
    img_pillow = Image.fromarray(image)
    # img_pillow.show()
    tile_names = []
    col = list()
    row = list()

    with tqdm(total=len(coord),
              desc="Tiling image",
              bar_format="{l_bar}{bar} [ time left: {remaining} ]") as pbar:
        for gene_name, imagerow, imagecol in zip(coord["barcode"], coord["imagerow"], coord["imagecol"]):
            imagecol,imagerow = imagecol * scale,imagerow * scale
            col.append(imagecol)
            row.append(imagerow)
            imagerow_down = imagerow - crop_size / 2
            imagerow_up = imagerow + crop_size / 2
            imagecol_left = imagecol - crop_size / 2
            imagecol_right = imagecol + crop_size / 2
            tile = img_pillow.crop(
                (imagecol_left, imagerow_down, imagecol_right, imagerow_up))
            tile.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)  #####
            tile.resize((target_size, target_size))  ######
            # tile_name = str(gene_name) + "-" +str(imagecol) + "-" + str(imagerow) + "-" + str(crop_size)
            tile_name = str(gene_name) + "-" + str(crop_size)
            out_tile = Path(save_path) / (tile_name + ".png")
            tile_names.append(str(out_tile))
            if verbose:
                print(
                    "generate tile at location ({}, {})".format(
                        str(imagecol), str(imagerow)))
            tile.save(out_tile, "PNG")
            pbar.update(1)

def compute_HDST_image_similarity(image_path, patch_size, save_dir,coord_path,n_components,dim_reduction="pca"):
    """
    HDST的特征相似性计算
    Args:
        image_path: 图像位置
        patch_size: patch大小
        save_dir: 保存路径
        coord_path: 坐标路径
        n_components: 使用PCA降维，降得维度
        dim_reduction: 使用降维的类型，默认为PCA

    Returns:提取的特征

    """
    if coord_path.endswith('.tsv'):
        coord = pd.read_csv(coord_path,sep="\t")
        coord = coord.loc[:,~coord.columns.str.contains("Unname")]
    elif coord_path.endswith('.csv'):
        coord = pd.read_csv(coord_path, sep=",")
        coord = coord.loc[:, ~coord.columns.str.contains("Unname")]
    else:
        raise ValueError("Unsupported file format. Only .tsv and .csv files are supported.")
    coord = coord.astype(int)
    coord_x_y = []
    for i in range(len(coord)):
        coord_x_y.append(list(coord.iloc[i]))
    # 对图像进行切分并保存
    split_and_save_patch(image_path, patch_size, coord_x_y, save_dir)
    cos = np.zeros([len(coord), len(coord)])
    rho = np.zeros([len(coord), len(coord)])
    feature = [[]]
    for i in tqdm(range(len(coord)), desc="Add image feature"):
        ig_1 = RNet50(save_dir + f'patch_{i}.png')
        img_1 = ig_1.flatten()
        feature.append(img_1)
    fea = pd.DataFrame(feature)
    fea = fea.dropna()

    # 执行特征降维
    # if dim_reduction == "pca":
    #     pca = PCA(n_components=n_components)
    #     reduced_feature = pca.fit_transform(fea)
    # elif dim_reduction == "umap":  # 如果使用 UMAP 进行降维
    #     umap_model = UMAP(n_components=n_components)
    #     reduced_feature = umap_model.fit_transform(fea)
    # else:
    #     raise ValueError("Unsupported dimensionality reduction method. Use 'pca' or 'umap'.")
    return fea

def compute_10X_image_similarity(image_path,save_path,coord_path,scale_path,n_components,dim_reduction="pca",crop_size=50):
    """
    10X的特征相似性计算
    Args:
        image_path: 图像的路径
        save_path: patch的保存路径
        coord_path: 坐标路径
        scale_path: 10X的缩放路径
        n_components: 使用PCA降维，降的维度信息
        dim_reduction: 使用降维的类型，默认为PCA
        crop_size: 10X中patch的保存与其他平台不同，crop_size保存在了patch的名字中

    Returns: 提取的特征

    """
    if coord_path.endswith('.tsv'):
        coord = pd.read_csv(coord_path, sep="\t",names=['cell_name','in_tissue','iamge_x','image_y','coord_x','coord_y'])
        coord.columns = ["barcode","in_tissue", "row", "col", "imagerow", "imagecol"]
        coord = coord.loc[:, ~coord.columns.str.contains("Unname")]
        coord = coord[coord['in_tissue'] == 1]
    elif coord_path.endswith('.csv'):
        coord = pd.read_csv(coord_path, sep=",",names=['cell_name','in_tissue','iamge_x','image_y','coord_x','coord_y'])
        coord.columns = ["barcode", "in_tissue", "row", "col", "imagerow", "imagecol"]
        coord = coord.loc[:, ~coord.columns.str.contains("Unname")]
        coord = coord[coord['in_tissue'] == 1]
    else:
        raise ValueError("Unsupported file format. Only .tsv and .csv files are supported.")
    scale_file = read_json(scale_path)
    if "hires" in image_path:
        scale = scale_file["tissue_hires_scalef"]
    elif "lowers" in image_path:
        scale = scale_file["tissue_lowers_scalef"]
    else:
        scale = 1

    image_crop(image_path, save_path, coord, scale,platform="10X")
    coord_row_col = coord[['imagerow', 'imagecol']].astype(float)
    coord_x_y = []
    for i in range(len(coord_row_col)):
        coord_x_y.append(list(coord_row_col.iloc[i]))
    cos = np.zeros([len(coord_row_col), len(coord_row_col)])
    rho = np.zeros([len(coord_row_col), len(coord_row_col)])
    feature = [[]]
    for i in tqdm(range(len(coord)), desc="Add image feature"):
        ig_1 = RNet50(save_path + str(coord["barcode"].iloc[i]) + "-" + str(crop_size) +f'.png')
        img_1 = ig_1.flatten()
        feature.append(img_1)
    fea = pd.DataFrame(feature)
    fea = fea.dropna()
    # 执行特征降维
    # if dim_reduction == "pca":
    #     pca = PCA(n_components=n_components)
    #     reduced_feature = pca.fit_transform(fea)
    # # elif dim_reduction == "umap":  # 如果使用 UMAP 进行降维
    # #     umap_model = UMAP(n_components=n_components)
    # #     reduced_feature = umap_model.fit_transform(fea)
    # else:
    #     raise ValueError("Unsupported dimensionality reduction method. Use 'pca' or 'umap'.")
    return fea



# 不同平台图像相似性计算
def feature_extract_extraction(save_path,coord_path,platform,n_components = None,image_path = None,scale_path = None):
    """
    不同平台图像相似性计算
    Args:
        save_path: 保存patch的路径
        coord_path: 坐标路径
        platform: 数据的平台
        n_components: PCA降维的维度
        image_path: 图像路径
        scale_path: 缩放文件路径，除10X外，其他平台为None

    Returns: 提取的特征

    """
    if platform == "HDST":
        patch_size = (64,64)
        fea = compute_HDST_image_similarity(image_path,patch_size,save_path,coord_path,n_components)
        return fea
    elif platform == "10X":
        fea = compute_10X_image_similarity(image_path, save_path, coord_path,scale_path,n_components)
        return fea
    else:
        raise ValueError("Unsupported platfrom.Only 'HDST','10X','slideseq','merfish'")

def compute_distance(coordinate_path,platform):
    """
    根据坐标信息，计算每个spot之间的距离
    Args:
        coordinate_path: 坐标路径
        platform: 平台

    Returns: 返回spot与spot之间的距离矩阵

    """
    # 使用欧氏距离计算spot之间的距离
    if platform == "10X":
        if coordinate_path.endswith('.tsv'):
            coordinate = pd.read_csv(coordinate_path, sep="\t",names=["barcode", "in_tissue", "row", "col", "imagerow", "imagecol"])
            coordinate.columns = ["barcode", "in_tissue", "row", "col", "imagerow", "imagecol"]
            coordinate = coordinate.loc[:, ~coordinate.columns.str.contains("Unname")]
            coordinate = coordinate[coordinate['in_tissue'] == 1]
        elif coordinate_path.endswith('.csv'):
            coordinate = pd.read_csv(coordinate_path, sep=",",names=["barcode", "in_tissue", "row", "col", "imagerow", "imagecol"])
            coordinate = coordinate.loc[:, ~coordinate.columns.str.contains("Unname")]
            coordinate.columns = ["barcode", "in_tissue", "row", "col", "imagerow", "imagecol"]
            coordinate = coordinate[coordinate['in_tissue'] == 1]
        else:
            raise ValueError("Unsupported file format. Only .tsv and .csv files are supported.")
        vector_X = np.array(coordinate['imagecol'])
        vector_Y = np.array(coordinate['imagerow'])
    elif platform == "HDST":
        if coordinate_path.endswith('.tsv'):
            coordinate = pd.read_csv(coordinate_path, sep="\t",index_col=0)
            coordinate = coordinate.loc[:, ~coordinate.columns.str.contains("Unname")]
        elif coordinate_path.endswith('.csv'):
            coordinate = pd.read_csv(coordinate_path, sep=",",index_col=0)
            coordinate = coordinate.loc[:, ~coordinate.columns.str.contains("Unname")]
        else:
            raise ValueError("Unsupported file format. Only .tsv and .csv files are supported.")
        coordinate.columns = ['segment_px_x','segment_px_y']
        vector_X = np.array(coordinate['segment_px_x'])
        vector_Y = np.array(coordinate['segment_px_y'])
    elif platform == "MERFISH":
        if coordinate_path.endswith('.tsv'):
            coordinate = pd.read_csv(coordinate_path, sep="\t")
            coordinate = coordinate.loc[:, ~coordinate.columns.str.contains("Unname")]
        elif coordinate_path.endswith('.csv'):
            coordinate = pd.read_csv(coordinate_path, sep=",")
            coordinate = coordinate.loc[:, ~coordinate.columns.str.contains("Unname")]
        else:
            raise ValueError("Unsupported file format. Only .tsv and .csv files are supported.")
        vector_X = np.array(coordinate['X'])
        vector_Y = np.array(coordinate['Y'])
    dimensionality = len(vector_X)
    matrix = np.zeros((dimensionality, dimensionality))
    for i in range(dimensionality):
        target = np.array([vector_X[i], vector_Y[i]])
        for j in range(dimensionality):
            if j > i:
                other = np.array([vector_X[j], vector_Y[j]])
                distance = np.sqrt(np.sum(np.square(target - other)))
                matrix[i][j] = matrix[j][i] = distance
    return matrix

def genarate_adj_matrix(coordinate_path,platform):
    """
    根据计算的spot间的距离，构造通讯网络，选择每个spot最近的五个相邻点
    Args:
        coordinate_path: 坐标路径
        platform: 平台

    Returns: 构建的网络

    """
    # 计算距离矩阵
    spot_distance = compute_distance(coordinate_path,platform)
    # 选择最小的五个距离
    min_distances = []
    for i in range(len(spot_distance)):
        min_distance_idx = sorted(
            [(val, idx) for idx, val in enumerate(spot_distance[i]) if (val != 0) and (100 < val < 300)])[:5]
        min_distances.append(min_distance_idx)

    # 生成邻接矩阵
    dimensionality = len(spot_distance)
    adj_matrix = np.zeros((dimensionality, dimensionality), dtype=int)
    for i, row in enumerate(min_distances):
        for _, c in row:
            adj_matrix[i][c] = 1
    return adj_matrix

import numpy as np
import scipy.sparse as sp

def sparse_to_tuple(sparse_mx):
    """
    将一个稀疏矩阵转换为(coords,values,shape)三元组，这个三元组可以用于创建Pytorch的稀疏张量
    Args:
        sparse_mx: 输入的稀疏矩阵，格式可以是任何Scipy支持的稀疏矩阵格式

    Returns: tuple三元组
        - coords: 稀疏矩阵中非零元素的坐标，形状为 (num_nonzero, 2)。
        - values: 稀疏矩阵中非零元素的值，形状为 (num_nonzero,)。
        - shape: 稀疏矩阵的形状 (rows, cols)，即矩阵的大小。

    """
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row,sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def preprocess_graph(adj):
    """
    对输入的邻接矩阵进行预处理，生成归一化的邻接矩阵并将其转换为三元组格式。
    该函数通常用于图神经网络中，以提高图卷积的稳定性和性能。
    Args:
        adj:输入的邻接矩阵，可以是稀疏矩阵或普通的二维数组

    Returns:返回归一化后的稀疏邻接矩阵的三元组表示 (coords, values, shape)。
        - coords: 非零元素的坐标，形状为 (num_nonzero, 2)。
        - values: 非零元素的值，形状为 (num_nonzero,)。
        - shape: 邻接矩阵的形状 (rows, cols)。

    """
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum,-0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)

def mask_test_edges(adj):
    """
    将图的边缘划分为训练集、验证集和测试集，并生成对应的假负边集（不存在的边）。
    此函数会移除对角元素（自环），随机划分 10% 的正样本为验证集，10% 的正样本为测试集，
    剩余的正样本用于训练，同时生成与正样本数量相同的假负样本。
    Args:
        adj:输入的邻接矩阵（稀疏矩阵格式）。

    Returns:
        tuple: 返回 (adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false)
        - adj_train: 用于训练的邻接矩阵（不包含验证集和测试集的边）。
        - train_edges: 训练集中的边（正样本）。
        - val_edges: 验证集中的边（正样本）。
        - val_edges_false: 验证集中的假负边（负样本）。
        - test_edges: 测试集中的边（正样本）。
        - test_edges_false: 测试集中的假负边（负样本）。

    """
    # Function to build test set with 10% positive links
    # NOTE：Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.
    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis,:],[0]),shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 10.))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges,np.hstack([test_edge_idx,val_edge_idx]),axis=0)

    def ismember(a,b,tol = 5):
        rows_close = np.all(np.round(a - b[:,None],tol)==0,axis=-1)
        return np.any(rows_close)

    test_edges_false = []
    while len(test_edges_false) <  len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) <  len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(val_edges_false, edges_all)
    assert ~ismember(val_edges, train_edges)
    assert ~ismember(test_edges, train_edges)
    assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T
    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false

def write_csv_matrix(matrix, filename, ifindex=False, ifheader=True, rownames=None, colnames=None, transpose=False):
    """
    将给定的矩阵保存为 CSV 文件，并可以选择是否保留行名、列名，以及是否转置矩阵。
    Args:
        matrix: 要保存的矩阵或二维数组。
        filename: 输出 CSV 文件的文件名，不需要扩展名 .csv，会自动添加。
        ifindex: 是否在 CSV 文件中包含行索引（默认 False，不包含索引）。
        ifheader: 是否在 CSV 文件中包含列名（默认 True，包含列名）。
        rownames: 行名列表，如果没有则默认为 None。
        colnames: 列名列表，如果没有则默认为 None。
        transpose: 是否在保存前对矩阵进行转置（默认 False，不转置）。

    Returns:函数不会返回任何值，只会将矩阵保存为 CSV 文件。
    """
    if transpose:
        matrix = matrix.T
        rownames, colnames = colnames, rownames
        ifindex, ifheader = ifheader, ifindex

    pd.DataFrame(matrix, index=rownames, columns=colnames).to_csv(filename+'.csv', index=ifindex, header=ifheader)
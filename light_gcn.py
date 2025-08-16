import torch
import numpy as np 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.data import DataLoader
import scipy.sparse as sp 
from collections import defaultdict
from tqdm import tqdm 

eps = 1e-8

def laplace_transform(graph):
    row_norm = np.sqrt(graph.sum(axis=1).A1) 
    row_norm = sp.diags(1/(row_norm+eps)) # [n_row, n_row] ~ [n_user, n_user]
    col_norm = np.sqrt(graph.sum(axis=0).A1) 
    col_norm = sp.diags(1/(col_norm+eps)) # [n_item, n_item]
    graph = row_norm @ graph @ col_norm
    return graph 

def to_tensor(graph):
    graph = graph.tocoo()
    values = graph.data 
    indices = np.vstack((graph.row, graph.col))
    graph = torch.sparse_coo_tensor(
        torch.LongTensor(indices),
        torch.FloatTensor(values),
        torch.Size(graph.shape)
    )
    return graph 

class light_gcn(nn.Module):
    def __init__(
        self, 
        n_user,
        n_item,
        dim,
        ui_graph, 
        conf
    ):
        super().__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.dim = dim
        self.ui_graph = ui_graph
        self.conf = conf 
        self.device = conf['device']
        # self.user_emb = nn.Embedding(self.n_user, embedding_dim=self.dim)
        # self.item_emb = nn.Embedding(self.n_item, embedding_dim=self.dim)
        # must init embedding
        # nn.init.xavier_uniform_(self.user_emb.weight)
        # nn.init.xavier_uniform_(self.item_emb.weight)

        self.user_emb = nn.Parameter(
            torch.FloatTensor(self.n_user, 64)
        )
        self.item_emb = nn.Parameter(
            torch.FloatTensor(self.n_item, 64)
        )
        nn.init.xavier_normal_(self.user_emb)
        nn.init.xavier_normal_(self.item_emb)

        self.adj_graph = self.get_graph()
        self.n_layer = 2 

    def get_graph(self):
        ui_graph = self.ui_graph 
        graph = sp.bmat(
            [[sp.csr_matrix((ui_graph.shape[0], ui_graph.shape[0])), ui_graph], [ui_graph.T, sp.csr_matrix((ui_graph.shape[1], ui_graph.shape[1]))]]
        )
        return to_tensor(laplace_transform(graph)).to(self.device)

    def propagate(self):
        features = torch.cat([self.user_emb, self.item_emb], dim=0)
        all_features = [features]
        for i in range(self.n_layer):
            features = torch.spmm(self.adj_graph, features)
            all_features.append(features)
        all_features = torch.stack(all_features, dim=0)
        all_features = torch.mean(all_features, dim=0)

        user_feat, item_feat = torch.split(
            tensor=all_features,
            split_size_or_sections=[self.n_user, self.n_item]
        )
        return user_feat, item_feat 

    def forward(self, user, pos_item, neg_item):
        user_feat, item_feat = self.propagate()
        # user, item: tensor or np
        return user_feat[user], item_feat[pos_item], item_feat[neg_item]
    
    def evaluate(self):
        return self.user_emb, self.item_emb

def sampling_data(train_data, num_neg=1):
    user, pos_item = train_data.nonzero()
    data = []
    for u, pos_i in zip(user, pos_item):
        for _ in range(num_neg):
            while True:
                a = np.random.choice([x for x in range(train_data.shape[1]) if x != pos_i])
                if train_data[u, a] == 0:
                    data.append((u, pos_i, a))
                    break 
    user, pos_item, neg_item = zip(*data)
    return (
        torch.tensor(user),
        torch.tensor(pos_item),
        torch.tensor(neg_item)
    )


def bpr_loss(user_emb, pos_item_emb, neg_item_emb):

    # print(f'user emb shape: {user_emb.shape}')
    # print(f'item emb shape: {pos_item_emb.shape}')

    # n_pair: num (user-item) pair 
    pos_score = torch.sum(user_emb*pos_item_emb, dim=-1) # [n_pair]
    neg_score = torch.sum(user_emb*neg_item_emb, dim=-1) # [n_pair]

    # print(f'shape pos score: {pos_score.shape}')
    # print(f'neg score shape: {neg_score.shape}')

    loss = -torch.mean(F.logsigmoid(pos_score - neg_score) + eps) 
    return loss  

def generate_synthetic_ui_data(n_user=100, n_item=300, min_items=5, max_items=10, seed=42):
    np.random.seed(seed)
    ui_data = []
    for user_id in range(n_user):
        n_interactions = np.random.randint(min_items, max_items + 1)
        interacted_items = np.random.choice(n_item, size=n_interactions, replace=False)
        for item_id in interacted_items:
            ui_data.append([user_id, item_id])
    return np.array(ui_data)

def to_binary_matrix(ui_array, n_user, n_item):
    mat = np.zeros((n_user, n_item), dtype=np.int32)
    for u, i in ui_array:
        mat[u, i] = 1
    return mat

def recall(output_topk, user_gt, item_gt):
    predicted_items = output_topk[user_gt]
    hits = (predicted_items == item_gt.unsqueeze(1)).any(dim=1).float()
    recall = hits.mean().item()
    return recall

def evaluate_recall(score, dataloader):
    # topk = [5,10,20,40,80]
    topk = [5,10,20]
    recall_list= defaultdict(list)

    user, pos_item, neg_item = dataloader
    for k in topk:
        value, idx = torch.topk(score, k=k)
        recall_k = recall(idx.to("cpu"), user, pos_item)
        recall_list[k].append(recall_k)

    return {k: np.mean(v) for k, v in recall_list.items()}

def evaluate(model, dataloader, k=20):
    u_emb, i_emb = model.evaluate()
    score = u_emb @ i_emb.T
    recall_ = evaluate_recall(score, dataloader)
    return recall_


# create dummy data 
# data = np.array([[1,0,1,0,1], [1,0,0,0,1], [0,0,0,1,1]])
n_user = 30
n_item = 50
data = generate_synthetic_ui_data(n_user=n_user, n_item=n_item, seed=1, min_items=10, max_items=20)
data = to_binary_matrix(data, n_user=n_user, n_item=n_item)

data_train = sampling_data(train_data=data, num_neg=1)
n_user, n_item = data.shape 
print(f'num user: {n_user}')
print(f'num item: {n_item}')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
conf = {
    'device': device, 
    'epoch': 30,
    'lr': 1e-1
}

model = light_gcn(
    n_user=n_user,
    n_item=n_item,
    dim=64,
    ui_graph=data,
    conf=conf 
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=conf['lr'])
# data_train = DataLoader(data_train, batch_size=5, shuffle=True)
print(f'len data train: {len(data_train)}')

training_bar = tqdm(range(conf['epoch']), total=conf['epoch'])
for epoch in training_bar:
    user, pos_item, neg_item = data_train
    user = user.to(device)
    pos_item = pos_item.to(device)
    neg_item = neg_item.to(device)

    user_emb, pos_item_emb, neg_item_emb = model(user, pos_item, neg_item)
    # cal bpr 
    loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f'loss: {loss}')
    recall_dict = (evaluate(model=model, dataloader=data_train))
    for k, v in recall_dict.items():
        print(f'recall@{k}: {v}')
    print('-' * 50)
    


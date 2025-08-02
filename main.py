import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

# ---------------------------
# 1. 设置随机种子，确保结果可复现
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------
# 2. 配置参数
class Config:
    def __init__(self):
        self.num_class = 2           # 分类数
        self.embedding_dim = 100     # 每个词的嵌入维度
        self.num_sentences = 30      # 每个样本包含的句子数
        self.num_words = 30          # 每个句子的词数

config = Config()

# ---------------------------
# 3. 定义数据集
# 加载文本、图结构和标签数据
# Dataset class
class NewsContentGraphDataset(Dataset):
    def __init__(self, content_path, graph_path, label_path):
        self.content = np.load(content_path)
        self.graph_feat = np.load(graph_path)
        self.labels = np.load(label_path)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        text = torch.tensor(self.content[idx], dtype=torch.float32)
        graph = torch.tensor(self.graph_feat[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return text, graph, label


# ---------------------------
# 4. 定义 HAN 模型（文本处理部分）
class SelfAttention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SelfAttention, self).__init__()
        self.W = nn.Linear(input_size, hidden_size, bias=True)
        self.u = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        u = torch.tanh(self.W(x))
        a = F.softmax(self.u(u), dim=1)  # (batch, seq_len, 1)
        weighted = a * x
        output = weighted.sum(dim=1)      # (batch, input_size)
        return output, a

class HAN(nn.Module):
    def __init__(self, config):
        super(HAN, self).__init__()
        self.embedding_dim = config.embedding_dim
        self.hidden_size_gru = 50
        self.hidden_size_att = 100
        # 第一层 GRU：针对每个句子内的词序列进行编码
        self.gru1 = nn.GRU(self.embedding_dim, self.hidden_size_gru,
                           bidirectional=True, batch_first=True)
        self.att1 = SelfAttention(self.hidden_size_gru * 2, self.hidden_size_att)
        # 第二层 GRU：针对句子级别进行编码
        self.gru2 = nn.GRU(self.hidden_size_att, self.hidden_size_gru,
                           bidirectional=True, batch_first=True)
        self.att2 = SelfAttention(self.hidden_size_gru * 2, self.hidden_size_att)
        # 分类层：将文档级表示映射到类别数
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size_att, config.num_class, bias=True)
        )
    
    def forward(self, x, return_feature=False):
        """
        输入 x 的形状：(batch, num_sentences, num_words, embedding_dim)
        1. 对每个句子内的词序列进行 GRU 编码和注意力加权，得到句子级表示
        2. 对所有句子的表示进行 GRU 编码和注意力加权，得到文档级表示
        3. 通过全连接层输出分类 logits
        若 return_feature 为 True，则返回文档级表示（供后续与图结构拼接使用）
        """
        batch_size, num_sentences, num_words, _ = x.size()
        # 将每个句子单独处理
        sentences = x.split(1, dim=1)  # 列表中每个元素形状：(batch, 1, num_words, embedding_dim)
        sentence_embeddings = []
        for s in sentences:
            s = s.squeeze(1)  # (batch, num_words, embedding_dim)
            gru_out, _ = self.gru1(s)  # (batch, num_words, hidden_size_gru*2)
            s_embed, _ = self.att1(gru_out)  # (batch, hidden_size_att)
            sentence_embeddings.append(s_embed.unsqueeze(1))
        # 拼接所有句子的表示 (batch, num_sentences, hidden_size_att)
        sentence_embeddings = torch.cat(sentence_embeddings, dim=1)
        gru_out, _ = self.gru2(sentence_embeddings)
        doc_embed, _ = self.att2(gru_out)  # (batch, hidden_size_att)
        logits = self.fc(doc_embed)        # (batch, num_class)
        if return_feature:
            return logits, doc_embed
        else:
            return logits

class VAE(nn.Module):
    def __init__(self, input_dim, z_dim=2):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, z_dim * 2)  # 输出均值和方差
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=-1)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return mu, logvar,recon_x

# 计算SKL
def compute_skl(mu1, logvar1, mu2, logvar2):
    kl1 = 0.6 * (logvar2 - logvar1 + (torch.exp(logvar1) + (mu1 - mu2)**2) / torch.exp(logvar2) - 1)
    kl2 = 0.4 * (logvar1 - logvar2 + (torch.exp(logvar2) + (mu2 - mu1)**2) / torch.exp(logvar1) - 1)
    skl = (kl1 + kl2) / 2
    return skl.mean(dim=1)

class AmbiguityModule(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, text_feat, graph_feat):
        # text_feat, graph_feat: (batch, feat_dim)
        combined = torch.cat([text_feat, graph_feat], dim=-1)  # (batch, 2*feat_dim)
        return self.net(combined).squeeze(-1)  # (batch,) ambiguity_score ∈ (0,1)

# 跨模态交互模块
class CrossModule(nn.Module):
    def __init__(self, text_dim, graph_dim, output_dim=64):
        super(CrossModule, self).__init__()
        self.text_proj = nn.Linear(text_dim, output_dim)
        self.graph_proj = nn.Linear(graph_dim, output_dim)
        self.attention = nn.Linear(output_dim * 2, 1)
    
    def forward(self, text_feat, graph_feat):
        text_proj = self.text_proj(text_feat)
        graph_proj = self.graph_proj(graph_feat)
        combined = torch.cat([text_proj, graph_proj], dim=-1)
        attn_scores = F.softmax(self.attention(combined), dim=0)
        cross_feat = attn_scores * text_proj + (1 - attn_scores) * graph_proj
        return cross_feat
    

class MultiModalSKL(nn.Module):
    def __init__(self, config, graph_dim=50, lambda_skl=0.5):
        super().__init__()
        self.han           = HAN(config)
        self.text_align    = nn.Linear(100, 64)
        self.graph_proj    = nn.Linear(graph_dim, 64)
        self.vae_text      = VAE(64)
        self.vae_graph     = VAE(64)
        self.cross_module  = CrossModule(64, 64)
        self.ambiguity_net = AmbiguityModule(64)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, config.num_class))
        
        # 正则权重
        self.lambda_skl    = lambda_skl

    def forward(self, text, graph):
        # 文本与图初始特征
        _, text_feat = self.han(text, return_feature=True)  # (batch,100)
        text_aligned = self.text_align(text_feat)           # (batch,64)
        graph_feat   = self.graph_proj(graph)               # (batch,64)

        # VAE 编码＋重构
        mu_t, logv_t, recon_t = self.vae_text(text_aligned)
        mu_g, logv_g, recon_g = self.vae_graph(graph_feat)
        # SKL 分数
        skl = compute_skl(mu_t, logv_t, mu_g, logv_g)       # (batch,)

        # 交叉特征
        cross_feat = self.cross_module(text_aligned, graph_feat)  # (batch,64)
        # 歧义打分
        ambiguity  = self.ambiguity_net(text_aligned, graph_feat) # (batch,)

        # 加权融合：高歧义偏文本，低歧义偏跨模态
        w_text = ambiguity.unsqueeze(1)
        w_cross = 1.0 - w_text
        text_final  = w_text  * text_aligned
        cross_final = w_cross * cross_feat

        fused = torch.cat([text_final, cross_final], dim=1)       # (batch,128)
        logits = self.classifier(fused)                           # (batch,C)

        # 把重构和 sk_loss 一并返回
        return logits, recon_t, recon_g, skl, ambiguity
# ---------------------------
# 训练和评估函数
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_loss_rec = 0.0
    total_loss_amb = 0.0
    num_batches = 0
    for texts, graphs, labels in dataloader:
        texts, graphs, labels = texts.to(device), graphs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits, recon_t, recon_g, skl, ambiguity = model(texts, graphs)

        loss_cls = criterion(logits, labels)

        loss_rec = F.mse_loss(recon_t, model.text_align(model.han(texts, True)[1])) \
                   + F.mse_loss(recon_g, model.graph_proj(graphs))

        loss_amb = F.mse_loss(ambiguity, skl)

        # 总 loss
        loss = loss_cls + 0.1 * loss_rec + model.lambda_skl * loss_amb
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * texts.size(0)
        total_loss_rec += loss_rec.item() * texts.size(0)
        total_loss_amb += loss_amb.item() * texts.size(0)
        num_batches += texts.size(0)

    avg_loss = total_loss / num_batches
    # avg_loss_rec = total_loss_rec / num_batches
    # avg_loss_amb = total_loss_amb / num_batches
    # print(f"Epoch Average -> loss_rec: {avg_loss_rec:.4f}  loss_amb: {avg_loss_amb:.4f}")
    # print(f"  skl mean={skl.mean():.4f}±{skl.std():.4f}, amb mean={ambiguity.mean():.4f}±{ambiguity.std():.4f}")

    return avg_loss

def evaluate(model, dataloader, criterion, device, rec_weight = 0.5):

    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for texts, graphs, labels in dataloader:
            texts, graphs, labels = texts.to(device), graphs.to(device), labels.to(device)

            # Forward pass
            logits, recon_t, recon_g, skl, ambiguity = model(texts, graphs)

            # 1) Classification loss
            loss_cls = criterion(logits, labels)

            # 2) VAE reconstruction loss
            # Recompute original aligned features
            _, text_feat = model.han(texts, return_feature=True)
            orig_text  = model.text_align(text_feat)
            orig_graph = model.graph_proj(graphs)
            loss_rec = F.mse_loss(recon_t, orig_text) + F.mse_loss(recon_g, orig_graph)

            # 3) Ambiguity regularization: encourage ambiguity ≈ skl
            loss_amb = F.mse_loss(ambiguity, skl)

            # Total loss
            loss = loss_cls + rec_weight * loss_rec + model.lambda_skl * loss_amb
            total_loss += loss.item() * texts.size(0)

            # Collect predictions
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss, all_labels, all_preds



# ---------------------------
# 7. 主函数：加载数据、训练和评估两种模型
if __name__ == '__main__':

    set_seed(42)

    batch_size = 16
    num_epochs = 50
    learning_rate = 1e-4 # gossip 0.001   

    content_path = 'path/to/your/content.npy'
    graph_path = 'path/to/your/graph_features.npy'
    label_path = 'path/to/your/labels.npy'

    # 加载数据集
    dataset = NewsContentGraphDataset(content_path, graph_path, label_path)
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()
    
    # 实例化模型
    config = Config()
    model_fusion = MultiModalSKL(config, lambda_skl=0.4).to(device)  
    optimizer_fusion = torch.optim.AdamW(model_fusion.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # 早停机制参数
    early_stop_patience = 8  
    best_val_loss = float('inf')
    no_improve_epochs = 0


    for epoch in range(num_epochs):
        train_loss = train_epoch(model_fusion, train_loader, criterion, optimizer_fusion, device)
        val_loss, true_labels_val, pred_labels_val = evaluate(model_fusion, val_loader, criterion, device)
        print(f"[融合] Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_epochs = 0

            best_model_state = model_fusion.state_dict()
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= early_stop_patience:
                print("Early stopping triggered!")
                break
    
    # 最终在测试集上评估模型性能
    final_loss, true_labels_test, pred_labels_test = evaluate(model_fusion, test_loader, criterion, device)
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
    acc = accuracy_score(true_labels_test, pred_labels_test)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels_test, pred_labels_test, average='binary')
    print("\n最终评估")
    print("Loss: {:.4f}".format(final_loss))
    print("Accuracy: {:.4f}".format(acc))
    print("Precision: {:.4f}".format(precision))
    print("Recall: {:.4f}".format(recall))
    print("F1 Score: {:.4f}".format(f1))
    print("Classification Report:")
    print(classification_report(true_labels_test, pred_labels_test, digits=4))
    


# loocv_gnn_optuna.py
import os
import json
import random
import scanpy as sc
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, accuracy_score
import random
import umap
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

import optuna
import argparse
from optuna.samplers import TPESampler

from utils import *
from models import MODEL_CLASSES

# ========== CLI Args ==========
parser = argparse.ArgumentParser()
parser.add_argument('--adata_dir', type=str)
parser.add_argument('--model', type=str, default='appnp', choices=['gcn','gat','sage','gin','appnp','recurrent','cheb','dna','tag','transformer','gated'])
parser.add_argument('--n_trials', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--num_epochs_search', type=int, default=10)
parser.add_argument('--num_epochs_eval', type=int, default=200)
parser.add_argument('--study_name', type=str, default='gnn_study_loocv')
parser.add_argument('--db_path', type=str, default='optuna_study_loocv.db')
args = parser.parse_args()

# ========== Configs ==========
selected_model = args.model
num_trials = args.n_trials
num_epochs_search = args.num_epochs_search
num_epochs_eval = args.num_epochs_eval
batch_size = args.batch_size

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

adata_dir = args.adata_dir
adata_paths = [os.path.join(adata_dir, x) for x in sorted(os.listdir(adata_dir)) if x.endswith('.h5ad')]
pnames = [os.path.splitext(os.path.basename(p))[0] for p in adata_paths]

for test_idx in range(len(adata_paths)):
    train_paths = [adata_paths[i] for i in range(len(adata_paths)) if i != test_idx]
    test_path = adata_paths[test_idx]
    test_name = pnames[test_idx]
    train_names = [pnames[i] for i in range(len(adata_paths)) if i != test_idx]

    results_dir = f'loocv_pid_{test_name}_{selected_model}'
    os.makedirs(results_dir, exist_ok=True)

    # Merge training adatas
    adatas = [sc.read_h5ad(p) for p in train_paths]
    for adata in adatas:
        adata.X = adata.X.toarray() if not isinstance(adata.X, np.ndarray) else adata.X
        adata.obs['label'] = LabelEncoder().fit_transform(adata.obs['Region'])
        construct_interaction_KNN(adata, n_neighbors=5)
    adata_train = adatas[0].concatenate(adatas[1:], batch_key="pid", batch_categories=train_names)
    X = adata_train.X
    Y = adata_train.obs['label'].values

    def evaluate(model, loader, loss_fn):
        model.eval()
        all_preds, all_labels, total_loss = [], [], 0
        with torch.no_grad():
            for batch in loader:
                batch = batch.to('cpu')
                out = model(batch.x, batch.edge_index)
                loss = loss_fn(out, batch.y)
                total_loss += loss.item() * batch.batch_size
                all_preds.append(out.argmax(dim=1))
                all_labels.append(batch.y)
        y_true = torch.cat(all_labels)
        y_pred = torch.cat(all_preds)
        acc = accuracy_score(y_true.numpy(), y_pred.numpy())
        return total_loss / len(loader.dataset), acc

    # ========== 3. Objective Function for Optuna ==========
    def objective(trial):
        hidden_channels = trial.suggest_categorical('hidden_channels', [64, 128, 256, 512, 1024])
        num_neighbors_sgraph = trial.suggest_categorical('num_neighbors', [3, 5, 15, 30])
        dropout = trial.suggest_float('dropout', 0.1, 0.6)
        lr = trial.suggest_float('lr', 1e-4, 1e-2)
        num_layers = trial.suggest_int('num_layers', 2, 5)
        optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'AdamW', 'SGD', 'RMSprop'])
        
        # Fixed code using spatial graph
        construct_interaction_KNN(adata_train, n_neighbors=num_neighbors_sgraph)  
        adj_spatial = adata_train.obsm['adj']
        edge_index = dense_to_sparse_edge_index(adj_spatial)

        train_idx, eval_idx = train_test_split(np.arange(len(Y)), test_size=0.2, random_state=42, stratify=Y)

        data = Data(x=torch.tensor(X, dtype=torch.float),
                    edge_index=edge_index,
                    y=torch.tensor(Y, dtype=torch.long))
        data.train_mask = torch.zeros(len(Y), dtype=torch.bool)
        data.train_mask[train_idx] = True
        data.eval_mask = torch.zeros(len(Y), dtype=torch.bool)
        data.eval_mask[eval_idx] = True

        model = MODEL_CLASSES[selected_model](
            in_channels=X.shape[1],
            hidden_channels=hidden_channels,
            out_channels=np.unique(Y).shape[0],
            num_layers=num_layers,
            dropout=dropout
        )

        if optimizer_name == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        elif optimizer_name == 'AdamW':
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        elif optimizer_name == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        elif optimizer_name == 'RMSprop':
            optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        loss_fn = nn.CrossEntropyLoss()

        train_loader = NeighborLoader(data, input_nodes=data.train_mask,
                                    num_neighbors=[num_neighbors_sgraph], batch_size=batch_size, shuffle=True, num_workers=0)
        eval_loader = NeighborLoader(data, input_nodes=data.eval_mask,
                                    num_neighbors=[num_neighbors_sgraph], batch_size=batch_size, shuffle=False, num_workers=0)

        for epoch in range(num_epochs_search):  
            model.train()
            for batch in train_loader:
                batch = batch.to('cpu')
                optimizer.zero_grad()
                out = model(batch.x, batch.edge_index)
                loss = loss_fn(out, batch.y)
                loss.backward()
                optimizer.step()

        _, eval_acc = evaluate(model, eval_loader, loss_fn)
        return eval_acc

    # ========== 4. Run Optuna ==========
    storage_str = f"sqlite:///{test_name}-{selected_model}-{args.db_path}"
    study = optuna.create_study(
        direction='maximize',
        study_name=args.study_name,
        storage=storage_str,
        load_if_exists=True,
        sampler=TPESampler(seed=42)
    )

    study.optimize(objective, n_trials=num_trials)


    # Save study result
    with open(os.path.join(results_dir, 'optuna_study.json'), 'w') as f:
        json.dump(study.best_trial.params, f, indent=2)

    # ========== 5. Final Training ==========
    best_params = study.best_trial.params
    config_str = f"{selected_model}_h{best_params['hidden_channels']}_l{best_params['num_layers']}_d{best_params['dropout']:.2f}_lr{best_params['lr']:.0e}"
    best_model_path = os.path.join(results_dir, f"{config_str}.pth")

    model = MODEL_CLASSES[selected_model](
        in_channels=X.shape[1],
        hidden_channels=best_params['hidden_channels'],
        out_channels=np.unique(Y).shape[0],
        num_layers=best_params['num_layers'],
        dropout=best_params['dropout']
    )
    print(model)
    optimizer_name = best_params.get('optimizer', 'AdamW')  # Default to AdamW if not present

    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'])
    elif optimizer_name == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=best_params['lr'])
    elif optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=best_params['lr'], momentum=0.9)
    elif optimizer_name == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=best_params['lr'])
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    # Fixed code using spatial graph
    construct_interaction_KNN(adata_train, n_neighbors=best_params['num_neighbors'])  
    adj_spatial = adata_train.obsm['adj']
    edge_index = dense_to_sparse_edge_index(adj_spatial)

    train_idx, eval_idx = train_test_split(np.arange(len(Y)), test_size=0.2, random_state=42, stratify=Y)

    data = Data(x=torch.tensor(X, dtype=torch.float),
                edge_index=edge_index,
                y=torch.tensor(Y, dtype=torch.long))
    data.train_mask = torch.zeros(len(Y), dtype=torch.bool)
    data.train_mask[train_idx] = True
    data.eval_mask = torch.zeros(len(Y), dtype=torch.bool)
    data.eval_mask[eval_idx] = True
    
    train_loader = NeighborLoader(data, input_nodes=data.train_mask,
                                num_neighbors=[best_params['num_neighbors']], batch_size=batch_size, shuffle=True, num_workers=0)
    eval_loader = NeighborLoader(data, input_nodes=data.eval_mask,
                                num_neighbors=[best_params['num_neighbors']], batch_size=batch_size, shuffle=False, num_workers=0)
    loss_fn = nn.CrossEntropyLoss()

    train_log = []
    best_eval_acc = 0

    for epoch in range(1, num_epochs_eval+1):
        model.train()
        for batch in train_loader:
            batch = batch.to('cpu')
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            loss = loss_fn(out, batch.y)
            loss.backward()
            optimizer.step()

        train_loss, train_acc = evaluate(model, train_loader, loss_fn)
        eval_loss, eval_acc = evaluate(model, eval_loader, loss_fn)

        if eval_acc > best_eval_acc:
            best_eval_acc = eval_acc
            torch.save(model.state_dict(), best_model_path)

        train_log.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'eval_loss': eval_loss,
            'eval_acc': eval_acc
        })

        print(f"Epoch {epoch:02d} | Train Acc: {train_acc:.4f} | Eval Acc: {eval_acc:.4f}")

    # Save logs
    with open(os.path.join(results_dir, 'train_log.json'), 'w') as f:
        json.dump(train_log, f, indent=2)

    with open(os.path.join(results_dir, 'best_model_config.json'), 'w') as f:
        json.dump({'best_model_path': best_model_path, 'params': best_params}, f, indent=2)

    # ========== 6. Test ==========
    with open(os.path.join(results_dir, 'best_model_config.json')) as f:
        best_model_meta = json.load(f)

    def run_clustering_eval(embeddings, labels, n_clusters):
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        pred = kmeans.fit_predict(embeddings.numpy())
        return float(adjusted_rand_score(labels, pred)), float(normalized_mutual_info_score(labels, pred))

    def extract_embeddings_with_hook(selected_model, model, data, device):
        if selected_model == 'gated':
            embeddings = []
            def hook(module, input, output):
                embeddings.append(output.detach().cpu())
            handle = model.lin_out.register_forward_hook(hook)
            model.eval()
            with torch.no_grad():
                _ = model(data.x.to(device), data.edge_index.to(device))
            handle.remove()
            return embeddings[0]  
        elif selected_model == 'appnp':
            embeddings = []
            def hook(module, input, output):
                embeddings.append(output.detach().cpu())
            handle = model.prop.register_forward_hook(hook)
            model.eval()
            with torch.no_grad():
                _ = model(data.x.to(device), data.edge_index.to(device))
            handle.remove()
            return embeddings[0]  
        elif selected_model == 'gcn' or 'sage':
            embeddings = []
            def hook(module, input, output):
                embeddings.append(output.detach().cpu())
            handle = model.convs[-1].register_forward_hook(hook)
            model.eval()
            with torch.no_grad():
                _ = model(data.x.to(device), data.edge_index.to(device))
            handle.remove()
            return embeddings[0]  
        elif selected_model == 'tag':
            embeddings = []
            def hook(module, input, output):
                embeddings.append(output.detach().cpu())
            handle = model.convs[-1].register_forward_hook(hook)
            model.eval()
            with torch.no_grad():
                _ = model(data.x.to(device), data.edge_index.to(device))
            handle.remove()
            return embeddings[0]  
        elif selected_model == 'gat':
            embeddings = []
            def hook(module, input, output):
                embeddings.append(output.detach().cpu())
            handle = model.convs[-1].register_forward_hook(hook)
            model.eval()
            with torch.no_grad():
                _ = model(data.x.to(device), data.edge_index.to(device))
            handle.remove()
            return embeddings[0]  
        elif selected_model == 'gin':
            embeddings = []
            def hook(module, input, output):
                embeddings.append(output.detach().cpu())
            handle = model.convs[-1][-1].register_forward_hook(hook)
            model.eval()
            with torch.no_grad():
                _ = model(data.x.to(device), data.edge_index.to(device))
            handle.remove()
            return embeddings[0]  
        elif selected_model == 'cheb':
            embeddings = []
            def hook(module, input, output):
                embeddings.append(output.detach().cpu())
            handle = model.convs[-1].register_forward_hook(hook)
            model.eval()
            with torch.no_grad():
                _ = model(data.x.to(device), data.edge_index.to(device))
            handle.remove()
            return embeddings[0]
        elif selected_model == 'transformer':
            embeddings = []
            def hook(module, input, output):
                embeddings.append(output.detach().cpu())
            handle = model.convs[-1].register_forward_hook(hook)
            model.eval()
            with torch.no_grad():
                _ = model(data.x.to(device), data.edge_index.to(device))
            handle.remove()
            return embeddings[0]

    model = MODEL_CLASSES[selected_model](
        in_channels=X.shape[1],
        hidden_channels=best_model_meta['params']['hidden_channels'],
        out_channels=np.unique(Y).shape[0],
        num_layers=best_model_meta['params']['num_layers'],
        dropout=best_model_meta['params']['dropout']
    )
    model.load_state_dict(torch.load(best_model_meta['best_model_path']))
    print(f"\nLoaded best model from {best_model_meta['best_model_path']}")

    # Set best num_neighbors for test
    best_num_neighbors_sgraph = best_model_meta['params']['num_neighbors']

    # Run test on the held-out sample
    test_pids = [test_idx]  # fixed: only test on the left-out pid in current LOOCV
    test_results = {}

    # Ensure loss_fn and device are defined
    loss_fn = nn.CrossEntropyLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    for pid in tqdm(test_pids, desc="Evaluating"):
        adata_test = sc.read_h5ad(adata_paths[pid])
        X_test = adata_test.X.toarray() if not isinstance(adata_test.X, np.ndarray) else adata_test.X
        Y_test = LabelEncoder().fit_transform(adata_test.obs['Region'])

        construct_interaction_KNN(adata_test, n_neighbors=best_num_neighbors_sgraph)
        adj_spatial_test = adata_test.obsm['adj']
        edge_index_test = dense_to_sparse_edge_index(adj_spatial_test)

        data_test = Data(x=torch.tensor(X_test, dtype=torch.float),
                        edge_index=edge_index_test,
                        y=torch.tensor(Y_test, dtype=torch.long))
        data_test.test_mask = torch.ones(data_test.num_nodes, dtype=torch.bool)

        test_loader = NeighborLoader(
            data_test, input_nodes=data_test.test_mask,
            num_neighbors=[best_num_neighbors_sgraph], batch_size=16, shuffle=False, num_workers=0)

        test_loss, test_acc = evaluate(model, test_loader, loss_fn)
        emb = extract_embeddings_with_hook(selected_model, model, data_test, device)
        ari, nmi = run_clustering_eval(emb, Y_test, n_clusters=len(np.unique(Y_test)))

        test_results[pid] = {'loss': float(test_loss), 'accuracy': float(test_acc), 'ARI': ari, 'NMI': nmi}
        print(f"Test PID {test_name} | Accuracy: {test_acc:.4f} | ARI: {ari:.4f} | NMI: {nmi:.4f}")

        df_all = pd.DataFrame([{
            'test_pid': test_name,
            'label': Y_test[i],
            **{f'emb_{j}': emb[i, j].item() for j in range(emb.shape[1])}
        } for i in range(emb.shape[0])])
        df_all.to_csv(os.path.join(results_dir, f'embeddings_pid_{test_name}.csv'), index=False)

        embedding_matrix = df_all[[col for col in df_all.columns if col.startswith("emb_")]].values
        df_all['UMAP_1'], df_all['UMAP_2'] = umap.UMAP(n_components=2).fit_transform(embedding_matrix).T
        df_all['TSNE_1'], df_all['TSNE_2'] = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(embedding_matrix).T

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

        # UMAP plot
        sns.scatterplot(
            data=df_all, x='UMAP_1', y='UMAP_2', hue='label',
            palette='tab10', s=15, linewidth=0, ax=axes[0], legend=False
        )
        axes[0].set_title("UMAP Projection", fontsize=14)
        axes[0].set_xlabel("UMAP-1")
        axes[0].set_ylabel("UMAP-2")

        # t-SNE plot
        sns.scatterplot(
            data=df_all, x='TSNE_1', y='TSNE_2', hue='label',
            palette='tab10', s=15, linewidth=0, ax=axes[1]
        )
        axes[1].set_title("t-SNE Projection", fontsize=14)
        axes[1].set_xlabel("t-SNE-1")
        axes[1].set_ylabel("t-SNE-2")

        handles, labels = axes[1].get_legend_handles_labels()
        axes[1].legend(handles=handles, labels=labels, title="Region", bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.savefig(os.path.join(results_dir, f'embedding_vis_pid_{test_name}.png'), dpi=600, bbox_inches='tight')
        plt.close()

    with open(os.path.join(results_dir, 'test_results.json'), 'w') as f:
        json.dump(test_results, f, indent=2)

    print("\n=== Final Test Accuracy per PID ===")
    for pid, res in test_results.items():
        print(f"PID {pid:02d} | Accuracy: {res['accuracy']:.4f} | ARI: {res['ARI']:.4f} | NMI: {res['NMI']:.4f}")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import random
import time

# ------------------ Model Definitions ------------------

class UserTower(nn.Module):
    def __init__(self, num_users, num_ages, num_genders, num_occupations, num_zipcodes,
                 num_contexts, history_embed_dim, embed_dim=32, hidden_dims=[128, 64]):
        super().__init__()
        self.user_id_emb = nn.Embedding(num_users, embed_dim)
        self.age_emb = nn.Embedding(num_ages, 8)
        self.gender_emb = nn.Embedding(num_genders, 4)
        self.occupation_emb = nn.Embedding(num_occupations, 8)
        self.zip_emb = nn.Embedding(num_zipcodes, 8)
        self.context_emb = nn.Embedding(num_contexts, 8)
        self.history_proj = nn.Linear(history_embed_dim, 16)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim + 8 + 4 + 8 + 8 + 8 + 16, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU()
        )

    def forward(self, user_id, age, gender, occupation, zipcode, context, history_emb):
        x = torch.cat([
            self.user_id_emb(user_id),
            self.age_emb(age),
            self.gender_emb(gender),
            self.occupation_emb(occupation),
            self.zip_emb(zipcode),
            self.context_emb(context),
            self.history_proj(history_emb)
        ], dim=-1)
        return self.mlp(x)

class ItemTower(nn.Module):
    def __init__(self, num_items, num_genres, num_years, num_contexts, embed_dim=32, hidden_dims=[128, 64]):
        super().__init__()
        self.item_id_emb = nn.Embedding(num_items, embed_dim)
        self.genre_emb = nn.Embedding(num_genres, 8)
        self.year_emb = nn.Embedding(num_years, 4)
        self.context_emb = nn.Embedding(num_contexts, 8)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim + 8 + 4 + 8, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU()
        )
    def forward(self, item_id, genre, year, context):
        x = torch.cat([
            self.item_id_emb(item_id),
            self.genre_emb(genre),
            self.year_emb(year),
            self.context_emb(context)
        ], dim=-1)
        return self.mlp(x)

class TwoTowerModel(nn.Module):
    def __init__(self, num_users, num_ages, num_genders, num_occupations, num_zipcodes,
                 num_items, num_genres, num_years, num_contexts, history_embed_dim=32, embed_dim=32, hidden_dims=[128, 64]):
        super().__init__()
        self.user_tower = UserTower(num_users, num_ages, num_genders, num_occupations, num_zipcodes,
                                   num_contexts, history_embed_dim, embed_dim, hidden_dims)
        self.item_tower = ItemTower(num_items, num_genres, num_years, num_contexts, embed_dim, hidden_dims)

    def forward(self, user_inputs, item_inputs, history_emb):
        user_id, age, gender, occupation, zipcode, context = user_inputs
        item_id, genre, year, item_context = item_inputs
        user_emb = self.user_tower(user_id, age, gender, occupation, zipcode, context, history_emb)
        item_emb = self.item_tower(item_id, genre, year, item_context)
        return (user_emb * item_emb).sum(dim=-1)

class Reranker(nn.Module):
    def __init__(self, user_feat_dim, item_feat_dim, context_dim, hidden_dims=[128, 64]):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(user_feat_dim + item_feat_dim + context_dim + 1, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 1)
        )
    def forward(self, user_feats, item_feats, context_feats, two_tower_score):
        x = torch.cat([user_feats, item_feats, context_feats, two_tower_score.unsqueeze(-1)], dim=-1)
        return self.mlp(x).squeeze(-1)

# ------------------ Feature Engineering ------------------

def extract_context_features(timestamps):
    hours = pd.to_datetime(timestamps, unit='s').dt.hour
    return pd.cut(hours, bins=[-1, 6, 12, 18, 24], labels=[0, 1, 2, 3]).astype(int)

def build_feature_encoders(users_df, items_df, ratings_df):
    user_encoder = LabelEncoder().fit(users_df['userId'])
    age_encoder = LabelEncoder().fit(users_df['age'])
    gender_encoder = LabelEncoder().fit(users_df['gender'])
    occupation_encoder = LabelEncoder().fit(users_df['occupation'])
    zip_encoder = LabelEncoder().fit(users_df['zip_code'])
    item_encoder = LabelEncoder().fit(items_df['movieId'])
    genre_encoder = LabelEncoder().fit(items_df['genre'])
    year_encoder = LabelEncoder().fit(items_df['year'])
    context_encoder = LabelEncoder().fit([0, 1, 2, 3])
    return dict(
        user=user_encoder, age=age_encoder, gender=gender_encoder,
        occupation=occupation_encoder, zip=zip_encoder,
        item=item_encoder, genre=genre_encoder, year=year_encoder,
        context=context_encoder
    )

def encode_features(row, encoders):
    return (
        encoders['user'].transform([row['userId']])[0],
        encoders['age'].transform([row['age']])[0],
        encoders['gender'].transform([row['gender']])[0],
        encoders['occupation'].transform([row['occupation']])[0],
        encoders['zip'].transform([row['zip_code']])[0],
        encoders['context'].transform([row['context']])[0]
    ), (
        encoders['item'].transform([row['movieId']])[0],
        encoders['genre'].transform([row['genre']])[0],
        encoders['year'].transform([row['year']])[0],
        encoders['context'].transform([row['context']])[0]
    )

def compute_user_history_embeddings(ratings_df, item_emb_matrix, item_encoder, history_length=5):
    user_hist_embs = {}
    for user_id, group in ratings_df.groupby('userId'):
        item_ids = group.sort_values('timestamp')['movieId'].values[-history_length:]
        item_idxs = item_encoder.transform(item_ids)
        hist_embs = item_emb_matrix[item_idxs]
        user_hist_embs[user_id] = hist_embs.mean(axis=0)
    return user_hist_embs

# ------------------ Dataset and Training ------------------

class TwoTowerDataset(Dataset):
    def __init__(self, ratings_df, users_df, items_df, encoders, user_hist_embs, num_negatives=4):
        self.ratings = ratings_df
        self.users = users_df.set_index('userId')
        self.items = items_df.set_index('movieId')
        self.encoders = encoders
        self.user_hist_embs = user_hist_embs
        self.num_negatives = num_negatives
        self.item_ids = items_df['movieId'].values

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        row = self.ratings.iloc[idx]
        user_id = row['userId']
        item_id = row['movieId']
        user_feats, item_feats = encode_features(
            {**row, **self.users.loc[user_id], **self.items.loc[item_id]}, self.encoders
        )
        history_emb = self.user_hist_embs.get(user_id, np.zeros(16, dtype=np.float32))
        pos = (user_feats, item_feats, history_emb, 1.0)
        negs = []
        for _ in range(self.num_negatives):
            neg_item_id = np.random.choice(self.item_ids)
            neg_item_feats = encode_features(
                {**row, **self.users.loc[user_id], **self.items.loc[neg_item_id]}, self.encoders
            )[1]
            negs.append((user_feats, neg_item_feats, history_emb, 0.0))
        samples = [pos] + negs
        return samples

def collate_fn(batch):
    flat = [sample for samples in batch for sample in samples]
    user_feats, item_feats, history_embs, labels = zip(*flat)
    user_feats = torch.LongTensor(user_feats)
    item_feats = torch.LongTensor(item_feats)
    history_embs = torch.FloatTensor(np.array(history_embs))
    labels = torch.FloatTensor(labels)
    return user_feats, item_feats, history_embs, labels

def train_two_tower(model, dataloader, optimizer, device='mps', epochs=5):
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    for epoch in range(epochs):
        total_loss = 0.0
        for user_feats, item_feats, history_embs, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            user_feats = user_feats.to(device)
            item_feats = item_feats.to(device)
            history_embs = history_embs.to(device)
            labels = labels.to(device)
            user_id, age, gender, occupation, zipcode, context = [user_feats[:, i] for i in range(6)]
            item_id, genre, year, item_context = [item_feats[:, i] for i in range(4)]
            user_emb = model.user_tower(user_id, age, gender, occupation, zipcode, context, history_embs)
            item_emb = model.item_tower(item_id, genre, year, item_context)
            logits = (user_emb * item_emb).sum(dim=-1)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss = {total_loss / len(dataloader):.4f}")

# ------------------ Candidate Generation & Reranking ------------------

def get_user_embedding(model, user_feats, history_emb, device='mps'):
    user_id, age, gender, occupation, zipcode, context = [torch.LongTensor([user_feats[i]]).to(device) for i in range(6)]
    history_emb = torch.FloatTensor(history_emb).unsqueeze(0).to(device)
    with torch.no_grad():
        user_emb = model.user_tower(user_id, age, gender, occupation, zipcode, context, history_emb)
    return user_emb.cpu().numpy().squeeze(0)

def get_item_embeddings(model, items_df, encoders, device='mps'):
    item_embs = []
    for _, row in items_df.iterrows():
        item_id, genre, year, context = encode_features(row, encoders)[1]
        item_id = torch.LongTensor([item_id]).to(device)
        genre = torch.LongTensor([genre]).to(device)
        year = torch.LongTensor([year]).to(device)
        context = torch.LongTensor([context]).to(device)
        with torch.no_grad():
            emb = model.item_tower(item_id, genre, year, context)
        item_embs.append(emb.cpu().numpy().squeeze(0))
    return np.stack(item_embs)

def generate_candidates(user_emb, item_emb_matrix, top_k=100):
    nbrs = NearestNeighbors(n_neighbors=top_k, metric='cosine').fit(item_emb_matrix)
    distances, indices = nbrs.kneighbors(user_emb.reshape(1, -1))
    return indices[0]

def rerank_candidates(reranker, user_feats, history_emb, candidate_items, model, encoders, items_df, device='mps'):
    rerank_scores = []
    user_id, age, gender, occupation, zipcode, context = [torch.LongTensor([user_feats[i]]).to(device) for i in range(6)]
    history_emb = torch.FloatTensor(history_emb).unsqueeze(0).to(device)
    with torch.no_grad():
        user_emb = model.user_tower(user_id, age, gender, occupation, zipcode, context, history_emb)
    for item_idx in candidate_items:
        row = items_df.iloc[item_idx]
        item_id, genre, year, item_context = encode_features(row, encoders)[1]
        item_id = torch.LongTensor([item_id]).to(device)
        genre = torch.LongTensor([genre]).to(device)
        year = torch.LongTensor([year]).to(device)
        item_context = torch.LongTensor([item_context]).to(device)
        with torch.no_grad():
            item_emb = model.item_tower(item_id, genre, year, item_context)
            two_tower_score = (user_emb * item_emb).sum(dim=-1)
        user_vec = torch.cat([user_emb, history_emb], dim=-1)
        item_vec = item_emb
        context_vec = item_context.float().unsqueeze(0)
        score = reranker(user_vec, item_vec, context_vec, two_tower_score)
        rerank_scores.append(score.item())
    return rerank_scores

# ------------------ Evaluation ------------------

def recall_at_k(recommended, ground_truth, k):
    return len(set(recommended[:k]) & set(ground_truth)) / min(k, len(ground_truth))

def ndcg_at_k(recommended, ground_truth, k):
    dcg = 0.0
    for i, rec in enumerate(recommended[:k]):
        if rec in ground_truth:
            dcg += 1 / np.log2(i + 2)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(ground_truth), k)))
    return dcg / idcg if idcg > 0 else 0.0

# ------------------ Main Script ------------------

if __name__ == "__main__":
    print("🚀 TWO-TOWER RECOMMENDER DEMO")
    print("="*50)

    # 1. Load MovieLens data
    print("\n📂 Loading MovieLens data...")
    ratings = pd.read_csv('../../01_fundamentals/01_movielens_exploration/ml-1m/ratings.dat',
                         sep='::',
                         names=['userId', 'movieId', 'rating', 'timestamp'],
                         engine='python',
                         encoding='latin-1')
    users = pd.read_csv('../../01_fundamentals/01_movielens_exploration/ml-1m/users.dat',
                       sep='::',
                       names=['userId', 'gender', 'age', 'occupation', 'zip_code'],
                       engine='python',
                       encoding='latin-1')
    movies = pd.read_csv('../../01_fundamentals/01_movielens_exploration/ml-1m/movies.dat',
                        sep='::',
                        names=['movieId', 'title', 'genre'],
                        engine='python',
                        encoding='latin-1')

    # 2. Preprocess and add context features
    print("🛠️  Preprocessing and feature engineering...")
    ratings['context'] = extract_context_features(ratings['timestamp'])
    movies['year'] = movies['title'].str.extract(r'\((\d{4})\)')[0].fillna(1998).astype(int)
    movies['genre'] = movies['genre'].str.split('|').str[0]
    movies['context'] = 0  # Default context for items

    # 3. Build encoders and user history embeddings
    encoders = build_feature_encoders(users, movies, ratings)
    dummy_item_emb_matrix = np.random.randn(len(movies), 32).astype(np.float32)
    user_hist_embs = compute_user_history_embeddings(ratings, dummy_item_emb_matrix, encoders['item'])

    # 4. Prepare dataset and dataloader
    dataset = TwoTowerDataset(ratings, users, movies, encoders, user_hist_embs)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, collate_fn=collate_fn)

    # 5. Initialize and train two-tower model
    model = TwoTowerModel(
        num_users=len(encoders['user'].classes_),
        num_ages=len(encoders['age'].classes_),
        num_genders=len(encoders['gender'].classes_),
        num_occupations=len(encoders['occupation'].classes_),
        num_zipcodes=len(encoders['zip'].classes_),
        num_items=len(encoders['item'].classes_),
        num_genres=len(encoders['genre'].classes_),
        num_years=len(encoders['year'].classes_),
        num_contexts=4,
        history_embed_dim=32,
        embed_dim=32,
        hidden_dims=[128, 64]
    )
    model = model.to('mps')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_two_tower(model, dataloader, optimizer, device='mps', epochs=5)

    # 6. Generate all item embeddings
    item_emb_matrix = get_item_embeddings(model, movies, encoders, device='mps')

    # 7. For a test user, get embedding and generate candidates
    test_user_row = users.iloc[0]
    test_user_feats = encode_features(test_user_row, encoders)[0]
    test_history_emb = user_hist_embs.get(test_user_row['userId'], np.zeros(16, dtype=np.float32))
    user_emb = get_user_embedding(model, test_user_feats, test_history_emb, device='mps')
    candidate_indices = generate_candidates(user_emb, item_emb_matrix, top_k=10)

    # 8. Rerank candidates
    reranker = Reranker(user_feat_dim=32+16, item_feat_dim=64, context_dim=1)
    rerank_scores = rerank_candidates(reranker, test_user_feats, test_history_emb, candidate_indices, model, encoders, movies, device='mps')

    # 9. Print top recommendations
    print("\n📋 Top recommendations (reranked):")
    for idx, score in zip(candidate_indices, rerank_scores):
        print(f"{movies.iloc[idx]['title']}: {score:.3f}")

    print("\n✅ Two-tower recommender demo completed!")


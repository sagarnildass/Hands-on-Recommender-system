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

# Optimize PyTorch for GPU
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(False)

# ------------------ Model Definitions ------------------


class UserTower(nn.Module):
    def __init__(
        self,
        num_users,
        num_ages,
        num_genders,
        num_occupations,
        num_zipcodes,
        num_contexts,
        history_embed_dim,
        embed_dim=32,
        hidden_dims=[128, 64],
    ):
        super().__init__()
        self.user_id_emb = nn.Embedding(num_users, embed_dim)
        self.age_emb = nn.Embedding(num_ages, 16)
        self.gender_emb = nn.Embedding(num_genders, 16)
        self.occupation_emb = nn.Embedding(num_occupations, 16)
        self.zip_emb = nn.Embedding(num_zipcodes, 16)
        self.context_emb = nn.Embedding(num_contexts, 16)
        self.history_proj = nn.Linear(history_embed_dim, 64)
        # Calculate input dim: embed_dim + 5*16 + 64
        input_dim = embed_dim + 5*16 + 64
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

    def forward(self, user_id, age, gender, occupation, zipcode, context, history_emb):
        x = torch.cat(
            [
                self.user_id_emb(user_id),
                self.age_emb(age),
                self.gender_emb(gender),
                self.occupation_emb(occupation),
                self.zip_emb(zipcode),
                self.context_emb(context),
                self.history_proj(history_emb),
            ],
            dim=-1,
        )
        return self.mlp(x)


class ItemTower(nn.Module):
    def __init__(
        self,
        num_items,
        num_genres,
        num_years,
        num_contexts,
        embed_dim=32,
        hidden_dims=[128, 64],
    ):
        super().__init__()
        self.item_id_emb = nn.Embedding(num_items, embed_dim)
        self.genre_emb = nn.Embedding(num_genres, 16)
        self.year_emb = nn.Embedding(num_years, 16)
        self.context_emb = nn.Embedding(num_contexts, 16)
        # Calculate input dim: embed_dim + 3*16
        input_dim = embed_dim + 3*16
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

    def forward(self, item_id, genre, year, context):
        x = torch.cat(
            [
                self.item_id_emb(item_id),
                self.genre_emb(genre),
                self.year_emb(year),
                self.context_emb(context),
            ],
            dim=-1,
        )
        return self.mlp(x)


class TwoTowerModel(nn.Module):
    def __init__(
        self,
        num_users,
        num_ages,
        num_genders,
        num_occupations,
        num_zipcodes,
        num_items,
        num_genres,
        num_years,
        num_contexts,
        history_embed_dim=32,
        embed_dim=32,
        hidden_dims=[128, 64],
    ):
        super().__init__()
        self.user_tower = UserTower(
            num_users,
            num_ages,
            num_genders,
            num_occupations,
            num_zipcodes,
            num_contexts,
            history_embed_dim,
            embed_dim,
            hidden_dims,
        )
        self.item_tower = ItemTower(
            num_items, num_genres, num_years, num_contexts, embed_dim, hidden_dims
        )

    def forward(self, user_inputs, item_inputs, history_emb):
        user_id, age, gender, occupation, zipcode, context = user_inputs
        item_id, genre, year, item_context = item_inputs
        user_emb = self.user_tower(
            user_id, age, gender, occupation, zipcode, context, history_emb
        )
        item_emb = self.item_tower(item_id, genre, year, item_context)
        return (user_emb * item_emb).sum(dim=-1)


class Reranker(nn.Module):
    def __init__(
        self, user_feat_dim, item_feat_dim, context_dim, hidden_dims=[128, 64]
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(user_feat_dim + item_feat_dim + context_dim + 1, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 1),
        )

    def forward(self, user_feats, item_feats, context_feats, two_tower_score):
        x = torch.cat(
            [user_feats, item_feats, context_feats, two_tower_score.unsqueeze(-1)],
            dim=-1,
        )
        return self.mlp(x).squeeze(-1)


# ------------------ Feature Engineering ------------------


def extract_context_features(timestamps):
    hours = pd.to_datetime(timestamps, unit="s").dt.hour
    return pd.cut(hours, bins=[-1, 6, 12, 18, 24], labels=[0, 1, 2, 3]).astype(int)


def build_feature_encoders(users_df, items_df, ratings_df):
    user_encoder = LabelEncoder().fit(users_df["userId"])
    age_encoder = LabelEncoder().fit(users_df["age"])
    gender_encoder = LabelEncoder().fit(users_df["gender"])
    occupation_encoder = LabelEncoder().fit(users_df["occupation"])
    zip_encoder = LabelEncoder().fit(users_df["zip_code"])
    item_encoder = LabelEncoder().fit(items_df["movieId"])
    genre_encoder = LabelEncoder().fit(items_df["genre"])
    year_encoder = LabelEncoder().fit(items_df["year"])
    context_encoder = LabelEncoder().fit([0, 1, 2, 3])
    
    # Create fast dict-based lookup tables
    return dict(
        user=user_encoder,
        age=age_encoder,
        gender=gender_encoder,
        occupation=occupation_encoder,
        zip=zip_encoder,
        item=item_encoder,
        genre=genre_encoder,
        year=year_encoder,
        context=context_encoder,
        # Lookup dicts for O(1) encoding
        user_to_idx={val: idx for idx, val in enumerate(user_encoder.classes_)},
        age_to_idx={val: idx for idx, val in enumerate(age_encoder.classes_)},
        gender_to_idx={val: idx for idx, val in enumerate(gender_encoder.classes_)},
        occupation_to_idx={val: idx for idx, val in enumerate(occupation_encoder.classes_)},
        zip_to_idx={val: idx for idx, val in enumerate(zip_encoder.classes_)},
        item_to_idx={val: idx for idx, val in enumerate(item_encoder.classes_)},
        genre_to_idx={val: idx for idx, val in enumerate(genre_encoder.classes_)},
        year_to_idx={val: idx for idx, val in enumerate(year_encoder.classes_)},
        context_to_idx={0: 0, 1: 1, 2: 2, 3: 3},
    )


def encode_features(row, encoders):
    user_feats = (
        encoders["user_to_idx"].get(row["userId"], 0),
        encoders["age_to_idx"].get(row["age"], 0),
        encoders["gender_to_idx"].get(row["gender"], 0),
        encoders["occupation_to_idx"].get(row["occupation"], 0),
        encoders["zip_to_idx"].get(row["zip_code"], 0),
        encoders["context_to_idx"].get(row["context"], 0),
    )
    item_feats = (
        encoders["item_to_idx"].get(row["movieId"], 0),
        encoders["genre_to_idx"].get(row["genre"], 0),
        encoders["year_to_idx"].get(row["year"], 0),
        encoders["context_to_idx"].get(row["context"], 0),
    )
    return user_feats, item_feats


def encode_item_features(row, encoders):
    """Encode only item features (for movies dataframe which has no user columns)"""
    return (
        encoders["item_to_idx"].get(row["movieId"], 0),
        encoders["genre_to_idx"].get(row["genre"], 0),
        encoders["year_to_idx"].get(row["year"], 0),
        encoders["context_to_idx"].get(row.get("context", 0), 0),
    )


def compute_user_history_embeddings(
    ratings_df, item_emb_matrix, item_encoder, history_length=5
):
    user_hist_embs = {}
    for user_id, group in ratings_df.groupby("userId"):
        item_ids = group.sort_values("timestamp")["movieId"].values[-history_length:]
        item_idxs = item_encoder.transform(item_ids)
        hist_embs = item_emb_matrix[item_idxs]
        user_hist_embs[user_id] = hist_embs.mean(axis=0)
    return user_hist_embs


# ------------------ Dataset and Training ------------------


class TwoTowerDataset(Dataset):
    def __init__(
        self, ratings_df, users_df, items_df, encoders, user_hist_embs, num_negatives=4
    ):
        self.ratings = ratings_df
        self.users = users_df.set_index("userId")
        self.items = items_df.set_index("movieId")
        self.encoders = encoders
        self.user_hist_embs = user_hist_embs
        self.num_negatives = num_negatives
        self.item_ids = items_df["movieId"].values

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        row = self.ratings.iloc[idx]
        user_id = row["userId"]
        item_id = row["movieId"]
        user_feats, item_feats = encode_features(
            {**row, **self.users.loc[user_id], **self.items.loc[item_id]}, self.encoders
        )
        history_emb = self.user_hist_embs.get(user_id, np.zeros(16, dtype=np.float32))
        pos = (user_feats, item_feats, history_emb, 1.0)
        negs = []
        for _ in range(self.num_negatives):
            neg_item_id = np.random.choice(self.item_ids)
            neg_item_feats = encode_features(
                {**row, **self.users.loc[user_id], **self.items.loc[neg_item_id]},
                self.encoders,
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


def train_two_tower(model, dataloader, optimizer, device="cuda", epochs=5, grad_accumulation_steps=2):
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler('cuda')
    
    for epoch in range(epochs):
        total_loss = 0.0
        accumulated_loss = 0.0
        
        for batch_idx, (user_feats, item_feats, history_embs, labels) in enumerate(tqdm(
            dataloader, desc=f"Epoch {epoch+1}"
        )):
            user_feats = user_feats.to(device, non_blocking=True)
            item_feats = item_feats.to(device, non_blocking=True)
            history_embs = history_embs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            user_id, age, gender, occupation, zipcode, context = [
                user_feats[:, i] for i in range(6)
            ]
            item_id, genre, year, item_context = [item_feats[:, i] for i in range(4)]
            
            with torch.amp.autocast('cuda'):
                user_emb = model.user_tower(
                    user_id, age, gender, occupation, zipcode, context, history_embs
                )
                item_emb = model.item_tower(item_id, genre, year, item_context)
                logits = (user_emb * item_emb).sum(dim=-1)
                loss = criterion(logits, labels) / grad_accumulation_steps
            
            scaler.scale(loss).backward()
            accumulated_loss += loss.item()
            
            if (batch_idx + 1) % grad_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                total_loss += accumulated_loss
                accumulated_loss = 0.0
        
        avg_loss = total_loss / (len(dataloader) // grad_accumulation_steps)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")


# ------------------ Candidate Generation & Reranking ------------------


def get_user_embedding(model, user_feats, history_emb, device="cuda"):
    user_id, age, gender, occupation, zipcode, context = [
        torch.LongTensor([user_feats[i]]).to(device) for i in range(6)
    ]
    history_emb = torch.FloatTensor(history_emb).unsqueeze(0).to(device)
    with torch.no_grad():
        user_emb = model.user_tower(
            user_id, age, gender, occupation, zipcode, context, history_emb
        )
    return user_emb.cpu().numpy().squeeze(0)


def get_item_embeddings(model, items_df, encoders, device="cuda", batch_size=256):
    """Batch encode all items for fast embedding generation"""
    item_embs = []
    
    with torch.no_grad():
        for batch_start in range(0, len(items_df), batch_size):
            batch_end = min(batch_start + batch_size, len(items_df))
            batch_items = items_df.iloc[batch_start:batch_end]
            
            # Encode all items in batch
            item_ids, genres, years, contexts = [], [], [], []
            for _, row in batch_items.iterrows():
                feats = encode_item_features(row, encoders)
                item_ids.append(feats[0])
                genres.append(feats[1])
                years.append(feats[2])
                contexts.append(feats[3])
            
            # Move to GPU as batches
            item_ids = torch.LongTensor(item_ids).to(device, non_blocking=True)
            genres = torch.LongTensor(genres).to(device, non_blocking=True)
            years = torch.LongTensor(years).to(device, non_blocking=True)
            contexts = torch.LongTensor(contexts).to(device, non_blocking=True)
            
            # Get embeddings for entire batch
            with torch.amp.autocast('cuda'):
                embs = model.item_tower(item_ids, genres, years, contexts)
            
            item_embs.append(embs.cpu().numpy())
    
    return np.vstack(item_embs)


def generate_candidates(user_emb, item_emb_matrix, top_k=100):
    nbrs = NearestNeighbors(n_neighbors=top_k, metric="cosine").fit(item_emb_matrix)
    distances, indices = nbrs.kneighbors(user_emb.reshape(1, -1))
    return indices[0]


def rerank_candidates(
    reranker,
    user_feats,
    history_emb,
    candidate_items,
    model,
    encoders,
    items_df,
    device="cuda",
):
    rerank_scores = []
    user_id, age, gender, occupation, zipcode, context = [
        torch.LongTensor([user_feats[i]]).to(device) for i in range(6)
    ]
    history_emb = torch.FloatTensor(history_emb).unsqueeze(0).to(device)
    with torch.no_grad():
        user_emb = model.user_tower(
            user_id, age, gender, occupation, zipcode, context, history_emb
        )
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
    print("=" * 50)

    # 1. Load MovieLens data
    print("\n📂 Loading MovieLens data...")
    ratings = pd.read_csv(
        "../../01_fundamentals/01_movielens_exploration/ml-1m/ratings.dat",
        sep="::",
        names=["userId", "movieId", "rating", "timestamp"],
        engine="python",
        encoding="latin-1",
    )
    users = pd.read_csv(
        "../../01_fundamentals/01_movielens_exploration/ml-1m/users.dat",
        sep="::",
        names=["userId", "gender", "age", "occupation", "zip_code"],
        engine="python",
        encoding="latin-1",
    )
    movies = pd.read_csv(
        "../../01_fundamentals/01_movielens_exploration/ml-1m/movies.dat",
        sep="::",
        names=["movieId", "title", "genre"],
        engine="python",
        encoding="latin-1",
    )

    # 2. Preprocess and add context features
    print("🛠️  Preprocessing and feature engineering...")
    ratings["context"] = extract_context_features(ratings["timestamp"])
    movies["year"] = (
        movies["title"].str.extract(r"\((\d{4})\)")[0].fillna(1998).astype(int)
    )
    movies["genre"] = movies["genre"].str.split("|").str[0]
    movies["context"] = 0  # Default context for items

    # 3. Build encoders and user history embeddings
    encoders = build_feature_encoders(users, movies, ratings)
    # Match the history_embed_dim=512 used in model
    dummy_item_emb_matrix = np.random.randn(len(movies), 512).astype(np.float32)
    user_hist_embs = compute_user_history_embeddings(
        ratings, dummy_item_emb_matrix, encoders["item"]
    )

    # 4. Prepare dataset and dataloader with GPU optimization - MAXIMIZE memory usage
    dataset = TwoTowerDataset(ratings, users, movies, encoders, user_hist_embs)
    dataloader = DataLoader(
        dataset,
        batch_size=4096,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=4,
    )

    # 5. Initialize and train two-tower model with MASSIVE capacity to use full 32GB
    model = TwoTowerModel(
        num_users=len(encoders["user"].classes_),
        num_ages=len(encoders["age"].classes_),
        num_genders=len(encoders["gender"].classes_),
        num_occupations=len(encoders["occupation"].classes_),
        num_zipcodes=len(encoders["zip"].classes_),
        num_items=len(encoders["item"].classes_),
        num_genres=len(encoders["genre"].classes_),
        num_years=len(encoders["year"].classes_),
        num_contexts=4,
        history_embed_dim=512,
        embed_dim=1024,
        hidden_dims=[8192, 4096],
    )
    model = model.to("cuda")
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_two_tower(model, dataloader, optimizer, device="cuda", epochs=5, grad_accumulation_steps=4)

    # 6. Generate all item embeddings
    item_emb_matrix = get_item_embeddings(model, movies, encoders, device="cuda")

    # 7. For a test user, get embedding and generate candidates
    test_user_row = users.iloc[0]
    test_user_feats = encode_features(test_user_row, encoders)[0]
    test_history_emb = user_hist_embs.get(
        test_user_row["userId"], np.zeros(16, dtype=np.float32)
    )
    user_emb = get_user_embedding(
        model, test_user_feats, test_history_emb, device="cuda"
    )
    candidate_indices = generate_candidates(user_emb, item_emb_matrix, top_k=10)

    # 8. Rerank candidates
    reranker = Reranker(user_feat_dim=32 + 16, item_feat_dim=64, context_dim=1)
    rerank_scores = rerank_candidates(
        reranker,
        test_user_feats,
        test_history_emb,
        candidate_indices,
        model,
        encoders,
        movies,
        device="cuda",
    )

    # 9. Print top recommendations
    print("\n📋 Top recommendations (reranked):")
    for idx, score in zip(candidate_indices, rerank_scores):
        print(f"{movies.iloc[idx]['title']}: {score:.3f}")

    print("\n✅ Two-tower recommender demo completed!")

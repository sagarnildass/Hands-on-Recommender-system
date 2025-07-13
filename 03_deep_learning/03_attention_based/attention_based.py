import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
from typing import List, Tuple, Dict, Optional


class AttentionLayer(nn.Module):
    """
    Multi-head attention layer for recommendation systems
    """

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(AttentionLayer, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask=None):
        batch_size, seq_len, embed_dim = x.size()

        # Linear transformations
        Q = (
            self.query(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        K = (
            self.key(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        V = (
            self.value(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim**0.5)

        # Apply mask if provided
        if mask is not None:
            # Expand mask to match attention scores shape [batch_size, num_heads, seq_len, seq_len]
            mask = mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]
            mask = mask.expand(
                -1, self.num_heads, seq_len, -1
            )  # [batch_size, num_heads, seq_len, seq_len]
            scores = scores.masked_fill(mask == 0, -1e9)

        # Softmax and dropout
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        context = (
            context.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        )

        # Output projection
        output = self.out_proj(context)

        return output, attention_weights


class AttentionBasedRecommender(nn.Module):
    """
    Attention-based recommendation system
    """

    def __init__(
        self,
        num_users,
        num_items,
        embed_dim=64,
        num_heads=8,
        num_layers=2,
        max_seq_len=50,
        dropout=0.1,
    ):
        super(AttentionBasedRecommender, self).__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        # Embeddings
        self.user_embedding = nn.Embedding(num_users, embed_dim)
        self.item_embedding = nn.Embedding(num_items, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)

        # Attention layers
        self.attention_layers = nn.ModuleList(
            [AttentionLayer(embed_dim, num_heads, dropout) for _ in range(num_layers)]
        )

        # Layer normalization
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(embed_dim) for _ in range(num_layers)]
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
        )

        # Output layers - More complex for better differentiation
        self.output_layer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid(),  # Constrain output to [0, 1] then scale to [0.5, 5.0]
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.1)

    def forward(self, user_ids, item_sequences, attention_mask=None):
        """
        Forward pass with attention mechanism

        Args:
            user_ids: User IDs [batch_size]
            item_sequences: Item sequences [batch_size, seq_len]
            attention_mask: Mask for padding [batch_size, seq_len]
        """
        batch_size, seq_len = item_sequences.size()

        # Get embeddings
        user_embeds = self.user_embedding(user_ids).unsqueeze(
            1
        )  # [batch_size, 1, embed_dim]
        item_embeds = self.item_embedding(
            item_sequences
        )  # [batch_size, seq_len, embed_dim]

        # Position embeddings
        positions = torch.arange(seq_len, device=item_sequences.device).unsqueeze(0)
        position_embeds = self.position_embedding(positions)  # [1, seq_len, embed_dim]

        # Combine embeddings
        x = item_embeds + position_embeds  # [batch_size, seq_len, embed_dim]

        # Apply attention layers
        attention_weights_list = []
        for i, (attention_layer, layer_norm) in enumerate(
            zip(self.attention_layers, self.layer_norms)
        ):
            # Self-attention
            attn_output, attn_weights = attention_layer(x, attention_mask)
            x = layer_norm(x + attn_output)  # Residual connection

            # Feed-forward
            ffn_output = self.ffn(x)
            x = layer_norm(x + ffn_output)  # Residual connection

            attention_weights_list.append(attn_weights)

        # Global average pooling
        if attention_mask is not None:
            # Mask out padding tokens
            mask_expanded = attention_mask.unsqueeze(-1).float()
            x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            x = x.mean(dim=1)  # [batch_size, embed_dim]

        # Add user embedding
        x = x + user_embeds.squeeze(1)  # [batch_size, embed_dim]

        # Output prediction
        output = self.output_layer(x)  # [batch_size, 1]

        # Scale from [0, 1] to [0.5, 5.0] for ratings
        output = 0.5 + 4.5 * output

        return output.squeeze(-1), attention_weights_list


class SequenceDataset(Dataset):
    """
    Dataset for sequence-based recommendation training
    """

    def __init__(self, user_sequences, ratings, max_seq_len=50):
        self.user_sequences = user_sequences
        self.ratings = ratings
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.user_sequences)

    def __getitem__(self, idx):
        user_id, item_seq, target_item, rating = self.user_sequences[idx]

        # Create sequence with target item
        full_seq = item_seq + [target_item]

        # Pad or truncate sequence
        if len(full_seq) > self.max_seq_len:
            full_seq = full_seq[-self.max_seq_len :]  # Keep most recent
        else:
            full_seq = full_seq + [0] * (
                self.max_seq_len - len(full_seq)
            )  # Pad with zeros

        # Create attention mask
        attention_mask = [1] * min(len(full_seq), self.max_seq_len) + [0] * max(
            0, self.max_seq_len - len(full_seq)
        )

        # Normalize rating to [0, 1] range
        normalized_rating = (rating - 0.5) / 4.5  # Scale from [0.5, 5.0] to [0, 1]

        return {
            "user_id": torch.LongTensor([user_id]),
            "item_sequence": torch.LongTensor(full_seq),
            "rating": torch.FloatTensor([normalized_rating]),
            "attention_mask": torch.LongTensor(attention_mask),
        }


def create_sequences(ratings_df, max_seq_len=50):
    """
    Create user-item sequences from ratings data
    """
    print("�� Creating user-item sequences...")

    # Sort by user and timestamp
    ratings_sorted = ratings_df.sort_values(["userId", "timestamp"])

    sequences = []
    user_sequences = {}

    for user_id in tqdm(ratings_sorted["userId"].unique(), desc="Processing users"):
        user_ratings = ratings_sorted[ratings_sorted["userId"] == user_id]

        if len(user_ratings) < 2:  # Skip users with only one rating
            continue

        # Create sequence of items
        item_sequence = user_ratings["movieId"].tolist()

        # Create training samples (predict next item rating)
        for i in range(1, len(item_sequence)):
            history = item_sequence[:i]
            target_item = item_sequence[i]
            target_rating = user_ratings.iloc[i]["rating"]

            if len(history) >= 1:  # At least one item in history
                sequences.append((user_id, history, target_item, target_rating))

    return sequences


def train_attention_model(
    model,
    train_loader,
    val_loader,
    num_epochs=30,
    learning_rate=0.001,
    device="mps",
    verbose=True,
    patience=5,
):
    """
    Train the attention-based recommender
    """
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    patience_counter = 0

    if verbose:
        print(f"�� Training attention-based model for {num_epochs} epochs...")
        print(f"📊 Training samples: {len(train_loader.dataset)}")
        print(f"�� Validation samples: {len(val_loader.dataset)}")

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0

        for batch in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False
        ):
            user_ids = batch["user_id"].squeeze().to(device)
            item_sequences = batch["item_sequence"].to(device)
            ratings = batch["rating"].squeeze().to(device)
            attention_mask = batch["attention_mask"].to(device)

            optimizer.zero_grad()

            predictions, _ = model(user_ids, item_sequences, attention_mask)
            loss = criterion(predictions, ratings)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(
                val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False
            ):
                user_ids = batch["user_id"].squeeze().to(device)
                item_sequences = batch["item_sequence"].to(device)
                ratings = batch["rating"].squeeze().to(device)
                attention_mask = batch["attention_mask"].to(device)

                predictions, _ = model(user_ids, item_sequences, attention_mask)
                loss = criterion(predictions, ratings)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # Learning rate scheduling
        scheduler.step(val_loss)

        if verbose:
            print(
                f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}"
            )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_attention_model.pth")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            if verbose:
                print(f"🛑 Early stopping at epoch {epoch+1} (patience: {patience})")
            break

    return train_losses, val_losses


def evaluate_attention_model(model, test_loader, device="mps"):
    """
    Evaluate the attention-based model
    """
    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            user_ids = batch["user_id"].squeeze().to(device)
            item_sequences = batch["item_sequence"].to(device)
            ratings = batch["rating"].squeeze().to(device)
            attention_mask = batch["attention_mask"].to(device)

            preds, _ = model(user_ids, item_sequences, attention_mask)

            predictions.extend(preds.cpu().numpy())
            actuals.extend(ratings.cpu().numpy())

    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mae = mean_absolute_error(actuals, predictions)

    return rmse, mae, predictions, actuals


def visualize_attention_weights(
    model, user_id, item_sequence, item_names, device="mps"
):
    """
    Visualize attention weights for a user's item sequence
    """
    model.eval()

    # Prepare input
    user_tensor = torch.LongTensor([user_id]).to(device)

    # Pad sequence to max_seq_len
    if len(item_sequence) > model.max_seq_len:
        padded_seq = item_sequence[-model.max_seq_len :]
    else:
        padded_seq = item_sequence + [0] * (model.max_seq_len - len(item_sequence))

    seq_tensor = torch.LongTensor([padded_seq]).to(device)
    mask_tensor = torch.LongTensor(
        [[1] * len(item_sequence) + [0] * (model.max_seq_len - len(item_sequence))]
    ).to(device)

    with torch.no_grad():
        _, attention_weights_list = model(user_tensor, seq_tensor, mask_tensor)

    # Plot attention weights for each layer
    num_layers = len(attention_weights_list)
    fig, axes = plt.subplots(1, num_layers, figsize=(5 * num_layers, 5))

    if num_layers == 1:
        axes = [axes]

    for i, attention_weights in enumerate(attention_weights_list):
        # Average across heads
        avg_attention = (
            attention_weights[0].mean(dim=0).cpu().numpy()
        )  # [seq_len, seq_len]

        # Plot heatmap
        sns.heatmap(
            avg_attention[: len(item_sequence), : len(item_sequence)],
            xticklabels=item_names,
            yticklabels=item_names,
            cmap="Blues",
            ax=axes[i],
        )
        axes[i].set_title(f"Layer {i+1} Attention Weights")
        axes[i].set_xlabel("Query Items")
        axes[i].set_ylabel("Key Items")

    plt.tight_layout()
    plt.savefig("attention_visualization.png", dpi=300, bbox_inches="tight")
    plt.show()


def generate_attention_recommendations(
    model,
    user_id,
    user_sequence,
    item_to_idx,
    idx_to_item,
    movies_df,
    n_recommendations=10,
    device="mps",
):
    """
    Generate recommendations using attention-based model
    """
    model.eval()

    # Convert sequence to indices
    seq_indices = [
        item_to_idx.get(item, 0) for item in user_sequence if item in item_to_idx
    ]

    if len(seq_indices) == 0:
        print("⚠️  No valid items in user sequence")
        return []

    # Pad sequence
    if len(seq_indices) > model.max_seq_len:
        seq_indices = seq_indices[-model.max_seq_len :]
    else:
        seq_indices = seq_indices + [0] * (model.max_seq_len - len(seq_indices))

    # Create attention mask
    attention_mask = [1] * min(len(user_sequence), model.max_seq_len) + [0] * max(
        0, model.max_seq_len - len(user_sequence)
    )

    # Get predictions for all items
    recommendations = []

    with torch.no_grad():
        user_tensor = torch.LongTensor([user_id]).to(device)
        seq_tensor = torch.LongTensor([seq_indices]).to(device)
        mask_tensor = torch.LongTensor([attention_mask]).to(device)

        for item_idx in tqdm(
            range(1, len(item_to_idx)), desc="Generating recommendations"
        ):
            # Create sequence with target item
            test_seq = seq_indices[:-1] + [item_idx]  # Replace last item with target

            test_tensor = torch.LongTensor([test_seq]).to(device)

            prediction, _ = model(user_tensor, test_tensor, mask_tensor)
            score = prediction.item()

            if item_idx in idx_to_item:
                item_id = idx_to_item[item_idx]
                movie_info = movies_df[movies_df["movieId"] == item_id]
                if not movie_info.empty:
                    title = movie_info.iloc[0]["title"]
                    recommendations.append((item_id, title, score))

    # Sort by score and return top recommendations
    recommendations.sort(key=lambda x: x[2], reverse=True)
    return recommendations[:n_recommendations]


def run_attention_demo():
    """
    Run a complete demonstration of Attention-Based Recommenders
    """
    print("🚀 ATTENTION-BASED RECOMMENDERS DEMO")
    print("=" * 50)

    # Set device
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device("mps")
    print(f"🖥️  Using device: {device}")

    # Load data
    print("\n📂 Loading MovieLens data...")
    ratings = pd.read_csv(
        "../../01_fundamentals/01_movielens_exploration/ml-1m/ratings.dat",
        sep="::",
        names=["userId", "movieId", "rating", "timestamp"],
        engine="python",
        encoding="latin-1",
    )
    movies = pd.read_csv(
        "../../01_fundamentals/01_movielens_exploration/ml-1m/movies.dat",
        sep="::",
        names=["movieId", "title", "genres"],
        engine="python",
        encoding="latin-1",
    )

    print(f"📊 Loaded {len(ratings)} ratings and {len(movies)} movies")

    # Use subset for faster training
    ratings_subset = ratings.sample(n=500000, random_state=42)
    print(f" Using subset: {len(ratings_subset)} ratings")

    # Create sequences
    sequences = create_sequences(ratings_subset, max_seq_len=20)
    print(f"📊 Created {len(sequences)} training sequences")

    # Create encoders
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()

    # Encode users and items
    all_users = list(set([seq[0] for seq in sequences]))
    all_items = list(set([item for seq in sequences for item in seq[1] + [seq[2]]]))

    user_encoder.fit(all_users)
    item_encoder.fit(all_items)

    # Convert sequences to encoded format
    encoded_sequences = []
    for user_id, history, target_item, target_rating in tqdm(
        sequences, desc="Encoding sequences"
    ):
        if target_item in item_encoder.classes_:
            encoded_user = user_encoder.transform([user_id])[0]
            encoded_history = [
                item_encoder.transform([item])[0]
                for item in history
                if item in item_encoder.classes_
            ]
            encoded_target = item_encoder.transform([target_item])[0]

            if len(encoded_history) > 0:
                encoded_sequences.append(
                    (encoded_user, encoded_history, encoded_target, target_rating)
                )

    print(f"📊 Encoded {len(encoded_sequences)} sequences")

    # Split data
    np.random.shuffle(encoded_sequences)
    split_idx = int(0.8 * len(encoded_sequences))
    val_split_idx = int(0.9 * len(encoded_sequences))

    train_sequences = encoded_sequences[:split_idx]
    val_sequences = encoded_sequences[split_idx:val_split_idx]
    test_sequences = encoded_sequences[val_split_idx:]

    print(f"📊 Data split:")
    print(f"   Training: {len(train_sequences)} sequences")
    print(f"   Validation: {len(val_sequences)} sequences")
    print(f"   Test: {len(test_sequences)} sequences")

    # Create datasets
    train_dataset = SequenceDataset(train_sequences, None, max_seq_len=20)
    val_dataset = SequenceDataset(val_sequences, None, max_seq_len=20)
    test_dataset = SequenceDataset(test_sequences, None, max_seq_len=20)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Initialize model
    num_users = len(user_encoder.classes_)
    num_items = len(item_encoder.classes_)

    model = AttentionBasedRecommender(
        num_users=num_users,
        num_items=num_items,
        embed_dim=64,
        num_heads=8,
        num_layers=2,
        max_seq_len=20,
        dropout=0.1,
    )

    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train model
    train_losses, val_losses = train_attention_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=25,
        learning_rate=0.0005,  # Lower learning rate for better convergence
        device=device,
        verbose=True,
        patience=5,  # More patience
    )

    # Evaluate model
    rmse, mae, predictions, actuals = evaluate_attention_model(
        model, test_loader, device
    )
    print(f"\n Evaluation Results:")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   MAE: {mae:.4f}")

    # Generate recommendations
    print("\n🎬 Generating attention-based recommendations...")

    # Create reverse mappings
    item_to_idx = {item: idx for idx, item in enumerate(item_encoder.classes_)}
    idx_to_item = {idx: item for item, idx in item_to_idx.items()}

    # Test with a sample user
    test_user_id = user_encoder.transform([train_sequences[0][0]])[0]
    test_user_sequence = train_sequences[0][1][:5]  # First 5 items

    recommendations = generate_attention_recommendations(
        model=model,
        user_id=test_user_id,
        user_sequence=test_user_sequence,
        item_to_idx=item_to_idx,
        idx_to_item=idx_to_item,
        movies_df=movies,
        n_recommendations=10,
        device=device,
    )

    print(f"\n📋 Top 10 recommendations for user {test_user_id}:")
    for i, (item_id, title, score) in enumerate(recommendations, 1):
        print(f"   {i:2d}. {title}: {score:.3f}")

    # Visualize attention weights
    print("\n🔍 Visualizing attention weights...")
    item_names = [
        movies[movies["movieId"] == idx_to_item[idx]]["title"].iloc[0]
        for idx in test_user_sequence
        if idx in idx_to_item
    ]

    visualize_attention_weights(
        model, test_user_id, test_user_sequence, item_names, device
    )

    print("\n✅ Attention-based recommender demo completed!")
    print("\n🎯 Key insights:")
    print("   - Attention mechanisms can focus on relevant items in sequences")
    print("   - Multi-head attention captures different types of relationships")
    print("   - Position embeddings help model understand item order")
    print("   - Attention weights provide interpretability")


if __name__ == "__main__":
    run_attention_demo()

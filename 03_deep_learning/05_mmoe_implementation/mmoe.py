import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class MMOE(nn.Module):
    def __init__(
        self,
        num_users,
        num_items,
        embedding_dim,
        num_experts,
        expert_hidden,
        num_tasks,
        tower_hidden,
    ):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        input_dim = 2 * embedding_dim

        # Experts: list of MLPs
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, expert_hidden),
                nn.ReLU(),
                nn.Linear(expert_hidden, expert_hidden),
                nn.ReLU(),
            )
            for _ in range(num_experts)
        ])

        # Gates: one per task
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, num_experts),
            ) for _ in range(num_tasks)
        ])

        # Task-specific towers
        self.towers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(expert_hidden, tower_hidden),
                nn.ReLU(),
                nn.Linear(tower_hidden, 1),
            ) for _ in range(num_tasks)
        ])
        
    def forward(self, user_idx, item_idx):
        # Embed user and item, concatenate
        user_emb = self.user_embedding(user_idx)
        item_emb = self.item_embedding(item_idx)
        x = torch.cat([user_emb, item_emb], dim=1)

        # Experts: Each produces (batch_size, expert_hidden)
        expert_outputs = [expert(x) for expert in self.experts]
        expert_stack = torch.stack(expert_outputs, dim=2) # (batch_size, expert_hidden, num_experts)

        # For each task: gate, combine experts, then tower
        outputs = []
        for i, (gate, tower) in enumerate(zip(self.gates, self.towers)):
            gate_weights = F.softmax(gate(x), dim=1) # (batch_size, num_experts)
            # Weighted sum of experts
            gate_weights = gate_weights.unsqueeze(1) # (batch_size, 1, num_experts)
            mmoe_out = torch.bmm(expert_stack, gate_weights.transpose(1,2)).squeeze(2)  # (batch_size, expert_hidden)
            out = tower(mmoe_out).squeeze(1)  # (batch_size,)
            outputs.append(out)
        return outputs

class MMOEDataset(Dataset):
    def __init__(self, ratings_df):
        self.user_idxs = ratings_df['user_idx'].values
        self.item_idxs = ratings_df['item_idx'].values
        self.rating_targets = ratings_df['rating_norm'].values  # Task 1: regression
        self.watch_targets = ratings_df['is_high_rating'].values  # Task 2: classification

    def __len__(self):
        return len(self.user_idxs)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.user_idxs[idx], dtype=torch.long),
            torch.tensor(self.item_idxs[idx], dtype=torch.long),
            torch.tensor(self.rating_targets[idx], dtype=torch.float),
            torch.tensor(self.watch_targets[idx], dtype=torch.float)
        )

def train_mmoe(model, train_loader, optimizer, device='cpu', epochs=10):
    model.train()
    model.to(device)
    
    # Loss functions for each task
    regression_loss = nn.MSELoss()  # For rating prediction
    classification_loss = nn.BCEWithLogitsLoss()  # For watch prediction
    
    for epoch in tqdm(range(epochs), desc="Training"):
        total_loss = 0
        for batch_idx, (user_idx, item_idx, rating_target, watch_target) in enumerate(train_loader):
            user_idx = user_idx.to(device)
            item_idx = item_idx.to(device)
            rating_target = rating_target.to(device)
            watch_target = watch_target.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass: get predictions for both tasks
            task1_pred, task2_pred = model(user_idx, item_idx)
            
            # Compute losses for each task
            loss1 = regression_loss(task1_pred, rating_target)
            loss2 = classification_loss(task2_pred, watch_target)
            
            # Combine losses (you can weight them if needed)
            total_task_loss = loss1 + loss2
            
            total_task_loss.backward()
            optimizer.step()
            
            total_loss += total_task_loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}: Loss = {total_task_loss.item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}: Average Loss = {avg_loss:.4f}")

def evaluate_mmoe(model, test_loader, device='cpu'):
    model.eval()
    model.to(device)
    
    all_rating_preds = []
    all_watch_preds = []
    all_rating_targets = []
    all_watch_targets = []
    
    with torch.no_grad():
        for user_idx, item_idx, rating_target, watch_target in tqdm(test_loader, desc="Evaluating"):
            user_idx = user_idx.to(device)
            item_idx = item_idx.to(device)
            
            # Get predictions
            task1_pred, task2_pred = model(user_idx, item_idx)
            
            # Store predictions and targets
            all_rating_preds.extend(task1_pred.cpu().numpy())
            all_watch_preds.extend(torch.sigmoid(task2_pred).cpu().numpy())  # Convert to probabilities
            all_rating_targets.extend(rating_target.numpy())
            all_watch_targets.extend(watch_target.numpy())
    
    # Convert to numpy arrays
    all_rating_preds = np.array(all_rating_preds)
    all_watch_preds = np.array(all_watch_preds)
    all_rating_targets = np.array(all_rating_targets)
    all_watch_targets = np.array(all_watch_targets)
    
    # Task 1: Rating prediction (regression)
    rmse = np.sqrt(mean_squared_error(all_rating_targets, all_rating_preds))
    mae = np.mean(np.abs(all_rating_targets - all_rating_preds))
    
    # Task 2: Watch prediction (classification)
    watch_pred_binary = (all_watch_preds > 0.5).astype(int)
    accuracy = accuracy_score(all_watch_targets, watch_pred_binary)
    auc = roc_auc_score(all_watch_targets, all_watch_preds)
    
    print(f"Task 1 (Rating Prediction):")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"\nTask 2 (Watch Prediction):")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  AUC: {auc:.4f}")
    
    return {
        'rating_rmse': rmse,
        'rating_mae': mae,
        'watch_accuracy': accuracy,
        'watch_auc': auc
    }

def analyze_gate_weights(model, test_loader, device='cpu'):
    """Analyze which experts each task prefers"""
    model.eval()
    model.to(device)
    
    task1_gate_weights = []
    task2_gate_weights = []
    
    with torch.no_grad():
        for user_idx, item_idx, _, _ in test_loader:
            user_idx = user_idx.to(device)
            item_idx = item_idx.to(device)
            
            # Get gate weights (we need to modify the forward pass to return gates)
            user_emb = model.user_embedding(user_idx)
            item_emb = model.item_embedding(item_idx)
            x = torch.cat([user_emb, item_emb], dim=1)
            
            gate1_weights = F.softmax(model.gates[0](x), dim=1)
            gate2_weights = F.softmax(model.gates[1](x), dim=1)
            
            task1_gate_weights.extend(gate1_weights.cpu().numpy())
            task2_gate_weights.extend(gate2_weights.cpu().numpy())
    
    task1_gate_weights = np.array(task1_gate_weights)
    task2_gate_weights = np.array(task2_gate_weights)
    
    # Average gate weights across all samples
    avg_gate1 = np.mean(task1_gate_weights, axis=0)
    avg_gate2 = np.mean(task2_gate_weights, axis=0)
    
    print(f"\nAverage Gate Weights:")
    print(f"Task 1 (Rating): {avg_gate1}")
    print(f"Task 2 (Watch): {avg_gate2}")
    
    # Plot gate weights
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.bar(range(len(avg_gate1)), avg_gate1)
    ax1.set_title('Task 1 (Rating) - Expert Usage')
    ax1.set_xlabel('Expert')
    ax1.set_ylabel('Average Weight')
    
    ax2.bar(range(len(avg_gate2)), avg_gate2)
    ax2.set_title('Task 2 (Watch) - Expert Usage')
    ax2.set_xlabel('Expert')
    ax2.set_ylabel('Average Weight')
    
    plt.tight_layout()
    plt.show()

def load_movielens_data():
    """Load MovieLens data (reuse from previous projects)"""
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
    users = pd.read_csv(
        "../../01_fundamentals/01_movielens_exploration/ml-1m/users.dat",
        sep="::",
        names=["userId", "gender", "age", "occupation", "zipcode"],
        engine="python",
        encoding="latin-1",
    )
    return ratings, movies, users

def generate_recommendations(model, user_idx, item2idx, movies_df, k=10, device='cpu'):
    """
    Generate top-k recommendations for a given user.
    
    Args:
        model: Trained MMOE model
        user_idx: User index
        item2idx: Mapping from movie ID to index
        movies_df: Movies dataframe with titles
        k: Number of recommendations to generate
        device: Device to run inference on
    
    Returns:
        List of (movie_id, title, rating_pred, watch_prob) tuples
    """
    model.eval()
    model.to(device)
    
    # Create user tensor (repeat for all items)
    user_tensor = torch.full((len(item2idx),), user_idx, dtype=torch.long, device=device)
    item_tensor = torch.arange(len(item2idx), dtype=torch.long, device=device)
    
    with torch.no_grad():
        # Get predictions for all items
        rating_preds, watch_logits = model(user_tensor, item_tensor)
        watch_probs = torch.sigmoid(watch_logits)
    
    # Convert to numpy
    rating_preds = rating_preds.cpu().numpy()
    watch_probs = watch_probs.cpu().numpy()
    
    # Create recommendation scores (combine both tasks)
    # You can adjust this weighting based on your preference
    recommendation_scores = 0.7 * rating_preds + 0.3 * watch_probs
    
    # Get top-k items
    top_indices = np.argsort(recommendation_scores)[::-1][:k]
    
    # Build recommendations list
    recommendations = []
    for idx in tqdm(top_indices, desc="Generating recommendations"):
        # Find the original movie ID
        movie_id = None
        for orig_id, item_idx in item2idx.items():
            if item_idx == idx:
                movie_id = orig_id
                break
        
        if movie_id is not None:
            # Get movie title
            movie_info = movies_df[movies_df['movieId'] == movie_id]
            title = movie_info['title'].iloc[0] if not movie_info.empty else f"Movie {movie_id}"
            
            recommendations.append({
                'movie_id': movie_id,
                'title': title,
                'rating_pred': float(rating_preds[idx]),
                'watch_prob': float(watch_probs[idx]),
                'combined_score': float(recommendation_scores[idx])
            })
    
    return recommendations

def print_recommendations(recommendations, user_id):
    """Pretty print recommendations"""
    print(f"\n🎬 Top 10 Recommendations for User {user_id}:")
    print("-" * 80)
    print(f"{'Rank':<4} {'Movie Title':<50} {'Rating':<8} {'Watch Prob':<10} {'Score':<8}")
    print("-" * 80)
    
    for i, rec in enumerate(recommendations, 1):
        print(f"{i:<4} {rec['title'][:48]:<50} {rec['rating_pred']:.3f}    {rec['watch_prob']:.3f}      {rec['combined_score']:.3f}")

if __name__ == "__main__":
    print("🚀 Starting MMOE Multi-Task Learning Experiment")
    print("=" * 50)
    
    # 1. Load and prepare data
    print("📂 Loading MovieLens data...")
    ratings, movies, users = load_movielens_data()
    
    # Encode user and item IDs
    user2idx = {uid: idx for idx, uid in enumerate(ratings['userId'].unique())}
    item2idx = {iid: idx for idx, iid in enumerate(ratings['movieId'].unique())}
    ratings['user_idx'] = ratings['userId'].map(user2idx)
    ratings['item_idx'] = ratings['movieId'].map(item2idx)
    
    # Create multi-task targets
    ratings['rating_norm'] = (ratings['rating'] - ratings['rating'].min()) / (ratings['rating'].max() - ratings['rating'].min())
    ratings['is_high_rating'] = (ratings['rating'] >= 4).astype(int)
    
    # Train/test split
    train_ratings, test_ratings = train_test_split(ratings, test_size=0.2, random_state=42)
    
    print(f"✅ Data loaded: {len(train_ratings)} train, {len(test_ratings)} test samples")
    print(f"📊 Users: {len(user2idx)}, Items: {len(item2idx)}")
    
    # 2. Create datasets and dataloaders
    print("\n🔄 Creating datasets...")
    train_dataset = MMOEDataset(train_ratings)
    test_dataset = MMOEDataset(test_ratings)
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    print(f"✅ Datasets created: {len(train_dataset)} train, {len(test_dataset)} test samples")
    
    # 3. Initialize model
    print("\n🧠 Initializing MMOE model...")
    num_users = len(user2idx)
    num_items = len(item2idx)
    embedding_dim = 32
    num_experts = 3
    expert_hidden = 64
    num_tasks = 2
    tower_hidden = 32
    
    model = MMOE(num_users, num_items, embedding_dim, num_experts, expert_hidden, num_tasks, tower_hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print(f"✅ Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # 4. Train the model
    print("\n🎯 Training MMOE model...")
    train_mmoe(model, train_loader, optimizer, device='mps', epochs=10)
    
    # 5. Evaluate the model
    print("\n📊 Evaluating model performance...")
    results = evaluate_mmoe(model, test_loader, device='mps')
    
    # 6. Analyze gate weights
    print("\n🔍 Analyzing expert usage patterns...")
    analyze_gate_weights(model, test_loader, device='mps')
    
    # 7. Generate sample recommendations
    print("\n🎬 Generating sample recommendations...")
    
    # Generate recommendations for a few sample users
    sample_users = [0, 100, 500]  # Different user types
    
    for user_id in sample_users:
        recommendations = generate_recommendations(
            model, user_id, item2idx, movies, k=10, device='mps'
        )
        print_recommendations(recommendations, user_id)
        print()
    
    print("\n🎉 MMOE experiment completed!")
    print("=" * 50)

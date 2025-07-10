import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, precision_score, recall_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class MovieLensDataset(Dataset):
    """
    Custom Dataset for MovieLens data
    """
    def __init__(self, user_ids, item_ids, ratings, transform=None):
        self.user_ids = torch.LongTensor(user_ids)
        self.item_ids = torch.LongTensor(item_ids)
        self.ratings = torch.FloatTensor(ratings)
        self.transform = transform

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        item_id = self.item_ids[idx]
        rating = self.ratings[idx]

        if self.transform:
            user_id, item_id, rating = self.transform(user_id, item_id, rating)

        return user_id, item_id, rating
    
class NeuralCollaborativeFiltering(nn.Module):
    """
    Neural Collaborative Filtering (NCF) model
    Combines matrix factorization with multi-layer perceptron
    """
    def __init__(self, num_users, num_items, embedding_dim=32, layers=[64, 32, 16], dropout=0.1):
        super(NeuralCollaborativeFiltering, self).__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim

        # User and item embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        # MLP layers
        self.mlp_layers = []
        input_dim = embedding_dim * 2 # Concatenated user and item embeddings

        for layer_size in layers:
            self.mlp_layers.extend([
                nn.Linear(input_dim, layer_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            input_dim = layer_size

        self.mlp = nn.Sequential(*self.mlp_layers)

        # Output layer
        self.output_layer = nn.Linear(layers[-1], 1)

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
    
    def forward(self, user_ids, item_ids):
        """
        Forward pass through the network
        """
        user_embeddings = self.user_embedding(user_ids)
        item_embeddings = self.item_embedding(item_ids)

        # Concatenate embeddings
        concatenated = torch.cat([user_embeddings, item_embeddings], dim=1)

        # Pass through MLP
        mlp_output = self.mlp(concatenated)

        # Pass through output layer
        output = self.output_layer(mlp_output)

        return output.squeeze()

    def train_model(self, train_loader, val_loader, num_epochs=50, learning_rate=0.001, 
                   device='cpu', verbose=True, patience=5):
        """
        Train the NCF model
        """

        self.to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0

        if verbose:
            print(f"�� Training NCF model for {num_epochs} epochs...")
            print(f"📊 Training samples: {len(train_loader.dataset)}")
            print(f"�� Validation samples: {len(val_loader.dataset)}")
        
        for epoch in tqdm(range(num_epochs), desc="Training epochs", disable=not verbose):
            # Training phase
            self.train()
            train_loss = 0.0

            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]") if verbose else train_loader

            for user_ids, item_ids, ratings in train_pbar:
                user_ids, item_ids, ratings = user_ids.to(device), item_ids.to(device), ratings.to(device)
                
                # Forward pass
                predictions = self(user_ids, item_ids)
                loss = criterion(predictions, ratings)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                if verbose and isinstance(train_pbar, tqdm):
                    train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)

            # Validation phase
            self.eval()
            val_loss = 0.0

            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]") if verbose else val_loader
                
                for user_ids, item_ids, ratings in val_pbar:
                    user_ids, item_ids, ratings = user_ids.to(device), item_ids.to(device), ratings.to(device)
                    
                    predictions = self(user_ids, item_ids)
                    loss = criterion(predictions, ratings)
                    val_loss += loss.item()
                    
                    if verbose and isinstance(val_pbar, tqdm):
                        val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                
                val_loss /= len(val_loader)
                val_losses.append(val_loss)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Save best model and check early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.state_dict(), 'best_ncf_model.pth')
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= patience:
                if verbose:
                    print(f"🛑 Early stopping at epoch {epoch+1} (patience: {patience})")
                break
            
            if verbose:
                print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        return train_losses, val_losses

    def predict(self, user_ids, item_ids, device='cpu'):
        """
        Make predictions for user-item pairs
        """
        self.eval()
        with torch.no_grad():
            user_ids = torch.LongTensor(user_ids).to(device)
            item_ids = torch.LongTensor(item_ids).to(device)
            predictions = self(user_ids, item_ids)
            return predictions.cpu().numpy()
    
    def get_embeddings(self, user_ids=None, item_ids=None, device='cpu'):
        """
        Get user and/or item embeddings
        """
        self.eval()
        with torch.no_grad():
            embeddings = {}

            if user_ids is not None:
                user_ids = torch.LongTensor(user_ids).to(device)
                user_embeddings = self.user_embedding(user_ids)
                embeddings['users'] = user_embeddings.cpu().numpy()

            if item_ids is not None:
                item_ids = torch.LongTensor(item_ids).to(device)
                item_embeddings = self.item_embedding(item_ids)
                embeddings['items'] = item_embeddings.cpu().numpy()

            return embeddings

    def evaluate_model(self, test_loader, device='cpu'):
        """
        Evaluate the model on test data
        """
        self.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for user_ids, item_ids, ratings in test_loader:
                user_ids, item_ids, ratings = user_ids.to(device), item_ids.to(device), ratings.to(device)
                
                preds = self(user_ids, item_ids)
                predictions.extend(preds.cpu().numpy())
                actuals.extend(ratings.cpu().numpy())
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Calculate metrics
        rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
        mae = np.mean(np.abs(predictions - actuals))
        
        # For binary classification (if using implicit feedback)
        if len(np.unique(actuals)) == 2:
            auc = roc_auc_score(actuals, predictions)
            precision = precision_score(actuals, predictions > 0.5)
            recall = recall_score(actuals, predictions > 0.5)
            
            return {
                'rmse': rmse,
                'mae': mae,
                'auc': auc,
                'precision': precision,
                'recall': recall
            }
        else:
            return {
                'rmse': rmse,
                'mae': mae
            }
    
def preprocess_data(ratings_df, test_size=0.2, val_size=0.1, random_state=42):
    """
    Preprocess MovieLens data for NCF training
    """
    print("�� Preprocessing data for NCF...")
    
    # Create label encoders for users and items
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    
    # Encode user and item IDs
    ratings_df['user_idx'] = user_encoder.fit_transform(ratings_df['userId'])
    ratings_df['item_idx'] = item_encoder.fit_transform(ratings_df['movieId'])
    
    # Normalize ratings to [0, 1] range
    ratings_df['rating_norm'] = (ratings_df['rating'] - ratings_df['rating'].min()) / \
                               (ratings_df['rating'].max() - ratings_df['rating'].min())
    
    # Split data
    train_val, test = train_test_split(ratings_df, test_size=test_size, random_state=random_state)
    train, val = train_test_split(train_val, test_size=val_size, random_state=random_state)
    
    print(f"📊 Data split:")
    print(f"   Training: {len(train)} samples")
    print(f"   Validation: {len(val)} samples")
    print(f"   Test: {len(test)} samples")
    print(f"   Users: {len(user_encoder.classes_)}")
    print(f"   Items: {len(item_encoder.classes_)}")
    
    return train, val, test, user_encoder, item_encoder

def create_data_loaders(train_data, val_data, test_data, batch_size=1024):
    """
    Create PyTorch DataLoaders for training, validation, and test
    """
    print("�� Creating data loaders...")
    
    # Create datasets
    train_dataset = MovieLensDataset(
        train_data['user_idx'].values,
        train_data['item_idx'].values,
        train_data['rating_norm'].values
    )
    
    val_dataset = MovieLensDataset(
        val_data['user_idx'].values,
        val_data['item_idx'].values,
        val_data['rating_norm'].values
    )
    
    test_dataset = MovieLensDataset(
        test_data['user_idx'].values,
        test_data['item_idx'].values,
        test_data['rating_norm'].values
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"✅ Data loaders created with batch size {batch_size}")
    
    return train_loader, val_loader, test_loader

def plot_training_history(train_losses, val_losses, save_path='training_history.png'):
    """
    Plot training and validation loss history
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('NCF Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def generate_recommendations(model, user_encoder, item_encoder, user_id, n_recommendations=10, 
                           device='cpu', exclude_rated=True, user_ratings=None):
    """
    Generate recommendations for a specific user
    """
    print(f"🎬 Generating recommendations for user {user_id}...")
    
    # Encode user ID
    if user_id not in user_encoder.classes_:
        print(f"⚠️  User {user_id} not found in training data")
        return []
    
    user_idx = user_encoder.transform([user_id])[0]
    
    # Get all items
    all_item_indices = np.arange(len(item_encoder.classes_))
    
    # Exclude already rated items if specified
    if exclude_rated and user_ratings is not None:
        rated_items = user_ratings[user_ratings['userId'] == user_id]['movieId'].values
        rated_indices = item_encoder.transform(rated_items)
        all_item_indices = np.setdiff1d(all_item_indices, rated_indices)
    
    # Make predictions
    user_ids = np.full(len(all_item_indices), user_idx)
    predictions = model.predict(user_ids, all_item_indices, device)
    
    # Get top recommendations
    top_indices = np.argsort(predictions)[::-1][:n_recommendations]
    top_items = item_encoder.inverse_transform(all_item_indices[top_indices])
    top_scores = predictions[top_indices]
    
    recommendations = list(zip(top_items, top_scores))
    
    return recommendations

def compare_with_baseline(ncf_results, baseline_results):
    """
    Compare NCF results with baseline methods
    """
    print("�� Comparing NCF with baseline methods...")
    
    comparison = {
        'NCF': ncf_results,
        'Baseline': baseline_results
    }
    
    print("\n🔍 Performance Comparison:")
    print("=" * 50)
    
    for metric in ['rmse', 'mae']:
        if metric in ncf_results and metric in baseline_results:
            ncf_score = ncf_results[metric]
            baseline_score = baseline_results[metric]
            improvement = ((baseline_score - ncf_score) / baseline_score) * 100
            
            print(f"{metric.upper()}:")
            print(f"   NCF: {ncf_score:.4f}")
            print(f"   Baseline: {baseline_score:.4f}")
            print(f"   Improvement: {improvement:.2f}%")
            print()
    
    return comparison

def run_ncf_demo():
    """
    Run a complete demonstration of Neural Collaborative Filtering
    """
    print("🚀 NEURAL COLLABORATIVE FILTERING DEMO")
    print("="*50)
    
    # Set device
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Using MPS for Apple Silicon
    device = torch.device("mps")
    print(f"🖥️  Using device: {device}")
    
    # Load data
    print("\n📂 Loading MovieLens data...")
    ratings = pd.read_csv('../../01_fundamentals/01_movielens_exploration/ml-1m/ratings.dat', 
                         sep='::', 
                         names=['userId', 'movieId', 'rating', 'timestamp'],
                         engine='python',
                         encoding='latin-1')
    
    movies = pd.read_csv('../../01_fundamentals/01_movielens_exploration/ml-1m/movies.dat', 
                        sep='::', 
                        names=['movieId', 'title', 'genres'],
                        engine='python',
                        encoding='latin-1')
    
    print(f"📊 Loaded {len(ratings)} ratings and {len(movies)} movies")
    
    # Use a subset for faster training
    ratings_subset = ratings.sample(n=min(500000, len(ratings)), random_state=42)
    print(f" Using subset: {len(ratings_subset)} ratings")
    
    # Preprocess data
    train_data, val_data, test_data, user_encoder, item_encoder = preprocess_data(ratings_subset)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(train_data, val_data, test_data, batch_size=512)
    
    # Initialize model
    print("\n🔧 Initializing NCF model...")
    num_users = len(user_encoder.classes_)
    num_items = len(item_encoder.classes_)
    
    model = NeuralCollaborativeFiltering(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=16,  # Reduced from 32
        layers=[32, 16, 8],  # Smaller layers
        dropout=0.3  # Increased dropout
    )
    
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    print("\n Training NCF model...")
    train_losses, val_losses = model.train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=20,  # Reduced epochs
        learning_rate=0.001,
        device=device,
        verbose=True,
        patience=3  # Early stopping patience
    )
    
    # Plot training history
    print("\n📈 Plotting training history...")
    plot_training_history(train_losses, val_losses)
    
    # Evaluate model
    print("\n📊 Evaluating model...")
    evaluation_results = model.evaluate_model(test_loader, device)
    
    print(f"\n Evaluation Results:")
    for metric, value in evaluation_results.items():
        print(f"   {metric.upper()}: {value:.4f}")
    
    # Generate recommendations
    print("\n🎬 Generating recommendations...")
    test_user = train_data['userId'].iloc[0]  # Use first user from training data
    
    recommendations = generate_recommendations(
        model=model,
        user_encoder=user_encoder,
        item_encoder=item_encoder,
        user_id=test_user,
        n_recommendations=10,
        device=device,
        exclude_rated=True,
        user_ratings=ratings_subset
    )
    
    print(f"\n📋 Top 10 recommendations for user {test_user}:")
    for i, (item_id, score) in enumerate(recommendations, 1):
        movie_title = movies[movies['movieId'] == item_id]['title'].iloc[0] if not movies[movies['movieId'] == item_id].empty else f"Movie {item_id}"
        print(f"   {i}. {movie_title}: {score:.3f}")
    
    # Analyze embeddings
    print("\n🔍 Analyzing learned embeddings...")
    user_embeddings = model.get_embeddings(user_ids=[user_encoder.transform([test_user])[0]], device=device)
    item_embeddings = model.get_embeddings(item_ids=[item_encoder.transform([recommendations[0][0]])[0]], device=device)
    
    print(f"   User embedding shape: {user_embeddings['users'].shape}")
    print(f"   Item embedding shape: {item_embeddings['items'].shape}")
    print(f"   User embedding norm: {np.linalg.norm(user_embeddings['users']):.4f}")
    print(f"   Item embedding norm: {np.linalg.norm(item_embeddings['items']):.4f}")
    
    # Test with different user types
    print("\n👥 Testing with different user types...")
    
    # Find users with different activity levels
    user_activity = ratings_subset.groupby('userId').size().sort_values(ascending=False)
    active_user = user_activity.index[0]  # Most active user
    inactive_user = user_activity.index[-1]  # Least active user
    
    for user_type, user_id in [("Active", active_user), ("Inactive", inactive_user)]:
        print(f"\n {user_type} user {user_id} recommendations:")
        user_recs = generate_recommendations(
            model=model,
            user_encoder=user_encoder,
            item_encoder=item_encoder,
            user_id=user_id,
            n_recommendations=5,
            device=device,
            exclude_rated=True,
            user_ratings=ratings_subset
        )
        
        for item_id, score in user_recs:
            movie_title = movies[movies['movieId'] == item_id]['title'].iloc[0] if not movies[movies['movieId'] == item_id].empty else f"Movie {item_id}"
            print(f"   {movie_title}: {score:.3f}")
    
    # Save model
    print("\n💾 Saving trained model...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'user_encoder': user_encoder,
        'item_encoder': item_encoder,
        'evaluation_results': evaluation_results,
        'model_config': {
            'num_users': num_users,
            'num_items': num_items,
            'embedding_dim': 32,
            'layers': [64, 32, 16]
        }
    }, 'ncf_model_complete.pth')
    
    print("✅ NCF demo completed!")
    print("\n🎯 Key insights:")
    print("   - Neural networks can learn complex user-item interactions")
    print("   - Embeddings capture latent user and item features")
    print("   - Deep learning can outperform traditional matrix factorization")
    print("   - Model can handle large-scale recommendation problems")


if __name__ == "__main__":
    run_ncf_demo()
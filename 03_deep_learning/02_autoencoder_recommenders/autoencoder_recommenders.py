import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class UserItemDataset(Dataset):
    """
    Dataset for user-item rating matrices
    """
    def __init__(self, user_item_matrix, noise_factor=0.1):
        # Convert DataFrame to numpy array if needed
        if hasattr(user_item_matrix, 'values'):
            matrix_values = user_item_matrix.values
        else:
            matrix_values = user_item_matrix
        self.user_item_matrix = torch.FloatTensor(matrix_values)
        self.noise_factor = noise_factor
        self.num_users, self.num_items = self.user_item_matrix.shape

    def __len__(self):
        return self.num_users

    def __getitem__(self, idx):
        # Get user's rating vector
        user_ratings = self.user_item_matrix[idx]

        # Create noisy input for denoising autoencoder
        if self.noise_factor > 0:
            noise = torch.randn_like(user_ratings) * self.noise_factor
            noisy_ratings = user_ratings + noise
            # Clip to valid rating range
            noisy_ratings = torch.clamp(noisy_ratings, 0, 5)
        else:
            noisy_ratings = user_ratings

        return noisy_ratings, user_ratings

class AutoencoderRecommender(nn.Module):
    """
    Autoencoder-based recommendation system
    """
    def __init__(self, num_items, hidden_dims=[128, 64, 32], dropout=0.2, 
                 activation='relu', use_sparse=False, sparsity_target=0.05):
        super(AutoencoderRecommender, self).__init__()

        self.num_items = num_items
        self.hidden_dims = hidden_dims
        self.use_sparse = use_sparse
        self.sparsity_target = sparsity_target

        # Encoder layers
        encoder_layers = []
        input_dim = num_items

        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                self._get_activation(activation),
                nn.Dropout(dropout)
            ])
            input_dim = hidden_dim

        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder layers
        decoder_layers = []
        hidden_dims_reversed = hidden_dims[::-1]

        for i, hidden_dim in enumerate(hidden_dims_reversed):
            if i == len(hidden_dims_reversed) - 1:
                # Final layer - output original dimension
                decoder_layers.extend([
                    nn.Linear(input_dim, num_items),
                    nn.Sigmoid()  # Output ratings in [0, 1] range
                ])
            else:
                decoder_layers.extend([
                    nn.Linear(input_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    self._get_activation(activation),
                    nn.Dropout(dropout)
                ])
            input_dim = hidden_dim

        self.decoder = nn.Sequential(*decoder_layers)

        # Initialize weights
        self._init_weights()

    def _get_activation(self, activation):
        """Get activation function"""
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'leaky_relu':
            return nn.LeakyReLU()
        else:
            return nn.ReLU()

    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        """Forward pass through autoencoder"""
        # Encode
        encoded = self.encoder(x)

        # Decode
        decoded = self.decoder(encoded)

        return decoded, encoded

    def get_latent_representation(self, x):
        """Get latent representation (encoded features)"""
        with torch.no_grad():
            return self.encoder(x)

    def train_model(self, train_loader, val_loader, num_epochs=50, learning_rate=0.001, 
                   device='mps', verbose=True, patience=5):
        """
        Train the autoencoder model
        """ 
        self.to(device)

        # Custom loss function for autoencoder
        def autoencoder_loss(reconstructed, original, encoded=None):
            # Reconstruction loss
            recon_loss = nn.MSELoss()(reconstructed, original)

            # Sparsity loss
            sparsity_loss = 0
            if self.use_sparse and encoded is not None:
                # KL divergence for sparsity
                sparsity_loss = torch.mean(
                    self.sparsity_target * torch.log(self.sparsity_target / (torch.mean(encoded, dim=0) + 1e-8)) +
                    (1 - self.sparsity_target) * torch.log((1 - self.sparsity_target) / (1 - torch.mean(encoded, dim=0) + 1e-8))
                )
            
            return recon_loss + 0.1 * sparsity_loss
        
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
        
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0

        if verbose:
            print(f"🎯 Training autoencoder for {num_epochs} epochs...")
            print(f"📊 Training samples: {len(train_loader.dataset)}")
            print(f" Validation samples: {len(val_loader.dataset)}")
        
        for epoch in range(num_epochs):
            # Training phase
            self.train()
            train_loss = 0.0
            
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]") if verbose else train_loader
            
            for noisy_ratings, original_ratings in train_pbar:
                noisy_ratings = noisy_ratings.to(device)
                original_ratings = original_ratings.to(device)
                
                # Forward pass
                reconstructed, encoded = self(noisy_ratings)
                loss = autoencoder_loss(reconstructed, original_ratings, encoded)
                
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
                
                for noisy_ratings, original_ratings in val_pbar:
                    noisy_ratings = noisy_ratings.to(device)
                    original_ratings = original_ratings.to(device)
                    
                    reconstructed, encoded = self(noisy_ratings)
                    loss = autoencoder_loss(reconstructed, original_ratings, encoded)
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
                torch.save(self.state_dict(), 'best_autoencoder_model.pth')
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
    
    def predict_ratings(self, user_ratings, device='mps'):
        """
        Predict ratings for a user
        """
        self.eval()
        with torch.no_grad():
            user_ratings = torch.FloatTensor(user_ratings).to(device)
            reconstructed, _ = self(user_ratings)
            return reconstructed.cpu().numpy()

    def get_user_embeddings(self, user_ratings, device='mps'):
        """
        Get latent embeddings for users
        """
        self.eval()
        with torch.no_grad():
            user_ratings = torch.FloatTensor(user_ratings).to(device)
            embeddings = self.get_latent_representation(user_ratings)
            return embeddings.cpu().numpy()
        
    def evaluate_model(self, test_loader, device='mps'):
        """
        Evaluate the autoencoder on test data
        """
        self.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for noisy_ratings, original_ratings in test_loader:
                noisy_ratings = noisy_ratings.to(device)
                original_ratings = original_ratings.to(device)
                
                reconstructed, _ = self(noisy_ratings)
                predictions.extend(reconstructed.cpu().numpy())
                actuals.extend(original_ratings.cpu().numpy())
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(actuals.flatten(), predictions.flatten()))
        mae = mean_absolute_error(actuals.flatten(), predictions.flatten())
        
        return {
            'rmse': rmse,
            'mae': mae,
            'predictions': predictions,
            'actuals': actuals
        }
    
def preprocess_data_for_autoencoder(ratings_df, test_size=0.2, val_size=0.1, random_state=42):
    """
    Preprocess data for autoencoder training
    """
    print(" Preprocessing data for autoencoder...")
    
    # Create user-item matrix
    user_item_matrix = ratings_df.pivot_table(
        index='userId', 
        columns='movieId', 
        values='rating', 
        fill_value=0
    )
    
    # Normalize ratings to [0, 1] range
    user_item_matrix = (user_item_matrix - user_item_matrix.min().min()) / \
                       (user_item_matrix.max().max() - user_item_matrix.min().min())
    
    # Split users into train/val/test
    user_ids = user_item_matrix.index
    train_users, test_users = train_test_split(user_ids, test_size=test_size, random_state=random_state)
    train_users, val_users = train_test_split(train_users, test_size=val_size, random_state=random_state)
    
    # Create splits
    train_matrix = user_item_matrix.loc[train_users]
    val_matrix = user_item_matrix.loc[val_users]
    test_matrix = user_item_matrix.loc[test_users]
    
    print(f"📊 Data split:")
    print(f"   Training: {len(train_matrix)} users")
    print(f"   Validation: {len(val_matrix)} users")
    print(f"   Test: {len(test_matrix)} users")
    print(f"   Items: {user_item_matrix.shape[1]}")
    
    return train_matrix, val_matrix, test_matrix, user_item_matrix

def create_autoencoder_data_loaders(train_matrix, val_matrix, test_matrix, batch_size=32, noise_factor=0.1):
    """
    Create data loaders for autoencoder training
    """
    print(" Creating data loaders...")
    
    # Create datasets
    train_dataset = UserItemDataset(train_matrix, noise_factor=noise_factor)
    val_dataset = UserItemDataset(val_matrix, noise_factor=0)  # No noise for validation
    test_dataset = UserItemDataset(test_matrix, noise_factor=0)  # No noise for testing
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"✅ Data loaders created with batch size {batch_size}")
    print(f"   Noise factor: {noise_factor}")
    
    return train_loader, val_loader, test_loader

def plot_training_history(train_losses, val_losses, save_path='autoencoder_training_history.png'):
    """
    Plot training and validation loss history
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Autoencoder Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def generate_autoencoder_recommendations(model, user_ratings, n_recommendations=10, 
                                       exclude_rated=True, device='mps'):
    """
    Generate recommendations using autoencoder
    """
    print("🎬 Generating autoencoder recommendations...")
    
    # Get predictions
    predictions = model.predict_ratings(user_ratings, device)
    
    # Create recommendation list
    recommendations = []
    for i, pred_rating in enumerate(predictions[0]):
        # Skip if user already rated this item
        if exclude_rated and user_ratings[0][i] > 0:
            continue
        
        recommendations.append((i, pred_rating))
    
    # Sort by predicted rating
    recommendations.sort(key=lambda x: x[1], reverse=True)
    
    return recommendations[:n_recommendations]

def analyze_latent_space(model, user_ratings, device='mps'):
    """
    Analyze the latent space learned by the autoencoder
    """
    print("🔍 Analyzing latent space...")
    
    # Get latent representations
    embeddings = model.get_user_embeddings(user_ratings, device)
    
    # Calculate statistics
    mean_embedding = np.mean(embeddings, axis=0)
    std_embedding = np.std(embeddings, axis=0)
    sparsity = np.mean(embeddings == 0)
    
    print(f"   Latent space statistics:")
    print(f"   - Mean activation: {np.mean(mean_embedding):.4f}")
    print(f"   - Std activation: {np.mean(std_embedding):.4f}")
    print(f"   - Sparsity: {sparsity:.4f}")
    print(f"   - Embedding shape: {embeddings.shape}")
    
    return embeddings

def compare_autoencoder_methods(models_dict, test_loader, device='mps'):
    """
    Compare different autoencoder configurations
    """
    print("🔍 Comparing autoencoder methods...")
    
    results = {}
    
    for model_name, model in models_dict.items():
        print(f"   Testing {model_name}...")
        evaluation = model.evaluate_model(test_loader, device)
        results[model_name] = evaluation
    
    # Print comparison
    print(f"\n📊 Autoencoder Comparison:")
    print("=" * 50)
    
    for model_name, result in results.items():
        print(f"{model_name}:")
        print(f"   RMSE: {result['rmse']:.4f}")
        print(f"   MAE: {result['mae']:.4f}")
        print()
    
    return results

def visualize_reconstructions(model, original_ratings, device='mps', n_examples=5):
    """
    Visualize original vs reconstructed ratings
    """
    print("📊 Visualizing reconstructions...")
    
    # Get predictions
    predictions = model.predict_ratings(original_ratings, device)
    
    # Plot examples
    fig, axes = plt.subplots(n_examples, 1, figsize=(12, 3*n_examples))
    
    for i in range(min(n_examples, len(original_ratings))):
        axes[i].plot(original_ratings[i], label='Original', alpha=0.7)
        axes[i].plot(predictions[i], label='Reconstructed', alpha=0.7)
        axes[i].set_title(f'User {i+1}: Original vs Reconstructed Ratings')
        axes[i].set_xlabel('Item Index')
        axes[i].set_ylabel('Rating')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('reconstructions.png', dpi=300, bbox_inches='tight')
    plt.show()

def run_autoencoder_demo():
    """
    Run a complete demonstration of Autoencoder Recommenders
    """
    print(" AUTOENCODER RECOMMENDERS DEMO")
    print("="*50)
    
    # Set device
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    train_matrix, val_matrix, test_matrix, full_matrix = preprocess_data_for_autoencoder(ratings_subset)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_autoencoder_data_loaders(
        train_matrix, val_matrix, test_matrix, batch_size=64, noise_factor=0.1
    )
    
    # Initialize different autoencoder models
    print("\n🔧 Initializing autoencoder models...")
    num_items = train_matrix.shape[1]
    
    models = {
        'Standard Autoencoder': AutoencoderRecommender(
            num_items=num_items,
            hidden_dims=[128, 64, 32],
            dropout=0.2,
            activation='relu',
            use_sparse=False
        ),
        'Sparse Autoencoder': AutoencoderRecommender(
            num_items=num_items,
            hidden_dims=[128, 64, 32],
            dropout=0.2,
            activation='relu',
            use_sparse=True,
            sparsity_target=0.05
        ),
        'Deep Autoencoder': AutoencoderRecommender(
            num_items=num_items,
            hidden_dims=[256, 128, 64, 32],
            dropout=0.3,
            activation='leaky_relu',
            use_sparse=False
        )
    }
    
    # Train models
    results = {}
    for model_name, model in models.items():
        print(f"\n Training {model_name}...")
        print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        train_losses, val_losses = model.train_model(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=25,
            learning_rate=0.001,
            device=device,
            verbose=True,
            patience=5
        )
        
        # Plot training history
        plot_training_history(train_losses, val_losses, f'{model_name.lower().replace(" ", "_")}_history.png')
        
        # Evaluate model
        evaluation = model.evaluate_model(test_loader, device)
        results[model_name] = evaluation
        
        print(f"✅ {model_name} training completed!")
        print(f"   RMSE: {evaluation['rmse']:.4f}")
        print(f"   MAE: {evaluation['mae']:.4f}")
    
    # Compare models
    print("\n📊 Model Comparison:")
    compare_autoencoder_methods(models, test_loader, device)
    
    # Generate recommendations with best model
    print("\n🎬 Generating recommendations...")
    best_model_name = min(results.keys(), key=lambda x: results[x]['rmse'])
    best_model = models[best_model_name]
    
    print(f"Using {best_model_name} for recommendations...")
    
    # Get a test user
    test_user_ratings = test_matrix.iloc[0:1].values
    test_user_id = test_matrix.index[0]
    
    # Generate recommendations
    recommendations = generate_autoencoder_recommendations(
        model=best_model,
        user_ratings=test_user_ratings,
        n_recommendations=10,
        exclude_rated=True,
        device=device
    )
    
    print(f"\n📋 Top 10 recommendations for user {test_user_id}:")
    for i, (item_idx, score) in enumerate(recommendations, 1):
        # Get movie ID from column index
        movie_id = test_matrix.columns[item_idx]
        movie_title = movies[movies['movieId'] == movie_id]['title'].iloc[0] if not movies[movies['movieId'] == movie_id].empty else f"Movie {movie_id}"
        print(f"   {i}. {movie_title}: {score:.3f}")
    
    # Analyze latent space
    print("\n🔍 Analyzing latent space...")
    embeddings = analyze_latent_space(best_model, test_user_ratings, device)
    
    # Visualize reconstructions
    print("\n📊 Visualizing reconstructions...")
    visualize_reconstructions(best_model, test_user_ratings, device, n_examples=3)
    
    # Test with different user types
    print("\n👥 Testing with different user types...")
    
    # Find users with different activity levels
    user_activity = ratings_subset.groupby('userId').size().sort_values(ascending=False)
    active_user_id = user_activity.index[0]
    inactive_user_id = user_activity.index[-1]
    
    for user_type, user_id in [("Active", active_user_id), ("Inactive", inactive_user_id)]:
        if user_id in full_matrix.index:
            print(f"\n {user_type} user {user_id} recommendations:")
            user_ratings = full_matrix.loc[user_id:user_id].values
            
            user_recs = generate_autoencoder_recommendations(
                model=best_model,
                user_ratings=user_ratings,
                n_recommendations=5,
                exclude_rated=True,
                device=device
            )
            
            for item_idx, score in user_recs:
                movie_id = full_matrix.columns[item_idx]
                movie_title = movies[movies['movieId'] == movie_id]['title'].iloc[0] if not movies[movies['movieId'] == movie_id].empty else f"Movie {movie_id}"
                print(f"   {movie_title}: {score:.3f}")
    
    # Save best model
    print("\n💾 Saving best model...")
    torch.save({
        'model_state_dict': best_model.state_dict(),
        'model_config': {
            'num_items': num_items,
            'hidden_dims': [128, 64, 32],
            'dropout': 0.2,
            'activation': 'relu'
        },
        'evaluation_results': results,
        'best_model_name': best_model_name
    }, 'best_autoencoder_model_complete.pth')
    
    print("✅ Autoencoder demo completed!")
    print("\n🎯 Key insights:")
    print("   - Autoencoders can learn compressed user representations")
    print("   - Denoising improves robustness to missing data")
    print("   - Sparse autoencoders can learn meaningful features")
    print("   - Deep architectures can capture complex patterns")


if __name__ == "__main__":
    run_autoencoder_demo()
    

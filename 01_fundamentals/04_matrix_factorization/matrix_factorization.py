import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")


def load_and_prepare_data():
    """
    Load MovieLens data and create rating matrix
    """
    # Load the data
    ratings = pd.read_csv(
        "../01_movielens_exploration/ml-1m/ratings.dat",
        sep="::",
        names=["userId", "movieId", "rating", "timestamp"],
        engine="python",
        encoding="latin-1",
    )

    movies = pd.read_csv(
        "../01_movielens_exploration/ml-1m/movies.dat",
        sep="::",
        names=["movieId", "title", "genres"],
        engine="python",
        encoding="latin-1",
    )

    # Create user-item matrix
    user_item_matrix = ratings.pivot_table(
        index="userId", columns="movieId", values="rating", fill_value=np.nan
    )

    print(f"User-item matrix shape: {user_item_matrix.shape}")
    print(
        f"Sparsity: {(1 - user_item_matrix.notna().sum().sum() / (user_item_matrix.shape[0] * user_item_matrix.shape[1])) * 100:.2f}%"
    )

    return user_item_matrix, movies, ratings


class FunkSVD:
    """
    FunkSVD implementation for matrix factorization
    """

    def __init__(self, n_factors=2, learning_rate=0.01, n_epochs=20, reg_param=0.1):
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.reg_param = reg_param
        self.user_factors = None
        self.item_factors = None
        self.user_biases = None
        self.item_biases = None
        self.global_mean = None

    def fit(self, ratings_matrix):
        """
        Train the FunkSVD model
        """
        print(f"Training FunkSVD with {self.n_factors} factors...")

        # Convert to sparse matrix
        ratings_sparse = csr_matrix(ratings_matrix.fillna(0))

        n_users, n_items = ratings_sparse.shape

        # Initialize factors randomly
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))

        # Initialize biases to 0
        self.user_biases = np.zeros(n_users)
        self.item_biases = np.zeros(n_items)
        self.global_mean = ratings_matrix.mean().mean()

        # Get non-zero ratings (actual ratings)
        user_indices, item_indices = ratings_sparse.nonzero()
        ratings_values = ratings_sparse.data

        print(f"Training on {len(ratings_values)} ratings...")

        # Training loop
        for epoch in tqdm(range(self.n_epochs), desc="Training"):
            total_error = 0

            for idx in tqdm(
                range(len(ratings_values)), desc="Processing ratings", leave=False
            ):
                user_idx = user_indices[idx]
                item_idx = item_indices[idx]
                actual_rating = ratings_values[idx]

                # Predicted rating
                predicted_rating = self._predict_single(user_idx, item_idx)

                # Calculate error
                error = actual_rating - predicted_rating
                total_error += error**2

                # Update factors using gradient descent
                self._update_factors(user_idx, item_idx, error)

            # Print progress
            if (epoch + 1) % 5 == 0:
                rmse = np.sqrt(total_error / len(ratings_values))
                print(f"Epoch {epoch + 1}/{self.n_epochs}, RMSE: {rmse:.4f}")

        print("Training completed!")

    def _predict_single(self, user_idx, item_idx):
        """
        Predict rating for a single user-item pair
        """
        prediction = (
            self.global_mean
            + self.user_biases[user_idx]
            + self.item_biases[item_idx]
            + np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
        )
        return prediction

    def _update_factors(self, user_idx, item_idx, error):
        """
        Update factors using gradient descent
        """
        # Update user factors
        self.user_factors[user_idx] += self.learning_rate * (
            error * self.item_factors[item_idx]
            - self.reg_param * self.user_factors[user_idx]
        )

        # Update item factors
        self.item_factors[item_idx] += self.learning_rate * (
            error * self.user_factors[user_idx]
            - self.reg_param * self.item_factors[item_idx]
        )

        # Update biases
        self.user_biases[user_idx] += self.learning_rate * (
            error - self.reg_param * self.user_biases[user_idx]
        )
        self.item_biases[item_idx] += self.learning_rate * (
            error - self.reg_param * self.item_biases[item_idx]
        )

    def predict(self, user_idx, item_idx):
        """
        Predict rating for user-item pair
        """
        return self._predict_single(user_idx, item_idx)


def evaluate_matrix_factorization(model, user_item_matrix, test_ratio=0.2):
    """
    Evaluate matrix factorization using train-test split
    """
    print("Evaluating matrix factorization...")

    # Create train-test split
    ratings_sparse = csr_matrix(user_item_matrix.fillna(0))
    user_indices, item_indices = ratings_sparse.nonzero()
    ratings_values = ratings_sparse.data

    # Random split
    np.random.seed(42)
    test_size = int(len(ratings_values) * test_ratio)
    test_indices = np.random.choice(len(ratings_values), test_size, replace=False)
    train_indices = np.setdiff1d(np.arange(len(ratings_values)), test_indices)

    # Calculate RMSE on test set
    test_errors = []
    for idx in tqdm(test_indices, desc="Processing test ratings", leave=False):
        user_idx = user_indices[idx]
        item_idx = item_indices[idx]
        actual_rating = ratings_values[idx]
        predicted_rating = model.predict(user_idx, item_idx)
        error = actual_rating - predicted_rating
        test_errors.append(error**2)

    rmse = np.sqrt(np.mean(test_errors))
    print(f"Test RMSE: {rmse:.4f}")

    return rmse


def get_matrix_factorization_recommendations(
    model, user_item_matrix, movies, user_id, n_recommendations=10
):
    """
    Get recommendations using matrix factorization
    """
    # Get user index
    user_idx = user_item_matrix.index.get_loc(user_id)

    # Get user's rated movies
    user_ratings = user_item_matrix.loc[user_id]
    rated_movies = user_ratings[user_ratings.notna()].index

    # Predict ratings for all movies
    predictions = []
    for movie_id in user_item_matrix.columns:
        if movie_id not in rated_movies:  # Only predict unrated movies
            movie_idx = user_item_matrix.columns.get_loc(movie_id)
            predicted_rating = model.predict(user_idx, movie_idx)
            predictions.append((movie_id, predicted_rating))

    # Sort by predicted rating
    predictions.sort(key=lambda x: x[1], reverse=True)

    # Get top recommendations
    top_recommendations = []
    for movie_id, predicted_rating in predictions[:n_recommendations]:
        movie_info = movies[movies["movieId"] == movie_id].iloc[0]
        top_recommendations.append(
            {
                "movieId": movie_id,
                "title": movie_info["title"],
                "genres": movie_info["genres"],
                "predicted_rating": predicted_rating,
            }
        )

    return pd.DataFrame(top_recommendations)


if __name__ == "__main__":
    # Load data
    print("Loading MovieLens data...")
    user_item_matrix, movies, ratings = load_and_prepare_data()

    # Train FunkSVD model
    print("\n" + "=" * 60)
    print("TRAINING MATRIX FACTORIZATION MODEL")
    print("=" * 60)

    model = FunkSVD(n_factors=20, learning_rate=0.01, n_epochs=20, reg_param=0.1)
    model.fit(user_item_matrix)

    # Evaluate model
    rmse = evaluate_matrix_factorization(model, user_item_matrix)

    # Test recommendations
    print(f"\n🎯 Testing recommendations...")

    # Find active users
    user_activity = user_item_matrix.notna().sum(axis=1)
    active_users = user_activity.nlargest(3).index

    for user_id in active_users:
        print(f"\n👤 User {user_id} (has {user_activity[user_id]} ratings):")

        recommendations = get_matrix_factorization_recommendations(
            model, user_item_matrix, movies, user_id, n_recommendations=5
        )

        print("Top 5 recommendations:")
        for i, rec in recommendations.iterrows():
            print(f"  {i+1}. {rec['title']} (Predicted: {rec['predicted_rating']:.2f})")
            print(f"     Genres: {rec['genres']}")

    print(f"\n✅ Matrix factorization completed!")
    print(f"📊 Final RMSE: {rmse:.4f}")

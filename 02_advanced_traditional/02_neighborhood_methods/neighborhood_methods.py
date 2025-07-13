import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")


def load_and_prepare_data():
    """
    Load MovieLens data for neighborhood methods
    """
    # Load the data
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

    # Create user-item matrix
    user_item_matrix = ratings.pivot_table(
        index="userId", columns="movieId", values="rating", fill_value=0
    )

    print(f"User-item matrix shape: {user_item_matrix.shape}")
    print(f"Non-zero ratings: {user_item_matrix.astype(bool).sum().sum()}")

    return user_item_matrix, movies, ratings


class KNNRecommender:
    """
    K-Nearest Neighbors Recommender
    """

    def __init__(self, n_neighbors=10, metric="cosine"):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.knn_model = None
        self.user_item_matrix = None
        self.user_similarities = None

    def fit(self, user_item_matrix):
        """
        Train KNN model
        """
        print(
            f"Training KNN with {self.n_neighbors} neighbors using {self.metric} similarity..."
        )
        start_time = time.time()

        self.user_item_matrix = user_item_matrix

        # Initialize KNN model
        self.knn_model = NearestNeighbors(
            n_neighbors=self.n_neighbors + 1,  # +1 for excluding self
            metric=self.metric,
            algorithm="brute",
        )

        # Fit the model
        self.knn_model.fit(user_item_matrix.values)

        training_time = time.time() - start_time
        print(f"KNN training completed in {training_time:.2f} seconds!")

    def get_recommendations(self, user_id, n_recommendations=10):
        """
        Get recommendations for a user using KNN
        """
        if user_id not in self.user_item_matrix.index:
            return []

        # Get user index
        user_idx = self.user_item_matrix.index.get_loc(user_id)

        # Find nearest neighbors
        user_vector = self.user_item_matrix.iloc[user_idx].values.reshape(
            1, -1
        )  # Use iloc, not loc
        distances, indices = self.knn_model.kneighbors(user_vector)

        # Get similar users (exclude self)
        similar_users = indices[0][1:]
        similar_distances = distances[0][1:]

        # Get user's rated movies
        user_ratings = self.user_item_matrix.iloc[user_idx]  # Use iloc, not loc
        rated_movies = user_ratings[user_ratings > 0].index

        # Collect recommendations from similar users
        recommendations = {}
        for i, similar_user_idx in enumerate(similar_users):
            similarity = 1 - similar_distances[i]  # Convert distance to similarity
            similar_user_ratings = self.user_item_matrix.iloc[similar_user_idx]

            # Get highly rated movies from similar user
            liked_movies = similar_user_ratings[similar_user_ratings >= 4].index

            for movie_id in liked_movies:
                if movie_id not in rated_movies:  # Don't recommend already rated
                    recommendations[movie_id] = (
                        recommendations.get(movie_id, 0) + similarity
                    )

        # Sort by score and return top recommendations
        sorted_recommendations = sorted(
            recommendations.items(), key=lambda x: x[1], reverse=True
        )
        return [
            movie_id for movie_id, score in sorted_recommendations[:n_recommendations]
        ]


class ClusteringRecommender:
    """
    Clustering-based recommender using K-means
    """

    def __init__(self, n_clusters=50):
        self.n_clusters = n_clusters
        self.kmeans_model = None
        self.user_item_matrix = None
        self.user_clusters = None

    def fit(self, user_item_matrix):
        """
        Train clustering model
        """
        print(f"Training clustering with {self.n_clusters} clusters...")
        start_time = time.time()

        self.user_item_matrix = user_item_matrix

        # Initialize K-means
        self.kmeans_model = KMeans(
            n_clusters=self.n_clusters, random_state=42, n_init=10
        )

        # Fit the model
        self.kmeans_model.fit(user_item_matrix.values)

        # Store cluster assignments
        self.user_clusters = self.kmeans_model.labels_

        training_time = time.time() - start_time
        print(f"Clustering training completed in {training_time:.2f} seconds!")

        # Print cluster statistics
        unique_clusters, counts = np.unique(self.user_clusters, return_counts=True)
        print(
            f"Cluster sizes: min={counts.min()}, max={counts.max()}, avg={counts.mean():.1f}"
        )

    def get_recommendations(self, user_id, n_recommendations=10):
        """
        Get recommendations for a user using clustering
        """
        if user_id not in self.user_item_matrix.index:
            return []

        # Get user index and cluster
        user_idx = self.user_item_matrix.index.get_loc(user_id)
        user_cluster = self.user_clusters[user_idx]

        # Find users in the same cluster
        cluster_users = np.where(self.user_clusters == user_cluster)[0]

        # Get user's rated movies
        user_ratings = self.user_item_matrix.iloc[user_idx]
        rated_movies = user_ratings[user_ratings > 0].index

        # Collect recommendations from cluster members
        recommendations = {}
        for cluster_user_idx in tqdm(
            cluster_users, desc="Processing cluster users", leave=False
        ):
            if cluster_user_idx != user_idx:  # Don't include self
                cluster_user_ratings = self.user_item_matrix.iloc[cluster_user_idx]

                # Get highly rated movies from cluster member
                liked_movies = cluster_user_ratings[cluster_user_ratings >= 4].index

                for movie_id in liked_movies:
                    if movie_id not in rated_movies:
                        recommendations[movie_id] = recommendations.get(movie_id, 0) + 1

        # Sort by score and return top recommendations
        sorted_recommendations = sorted(
            recommendations.items(), key=lambda x: x[1], reverse=True
        )
        return [
            movie_id for movie_id, score in sorted_recommendations[:n_recommendations]
        ]


def compare_neighborhood_methods(user_item_matrix, movies, ratings):
    """
    Compare different neighborhood methods
    """
    print("\n" + "=" * 60)
    print(" COMPARING NEIGHBORHOOD METHODS")
    print("=" * 60)

    # Test different methods
    methods = {
        "KNN (Cosine)": KNNRecommender(n_neighbors=20, metric="cosine"),
        "KNN (Euclidean)": KNNRecommender(n_neighbors=20, metric="euclidean"),
        "Clustering": ClusteringRecommender(n_clusters=50),
    }

    results = {}

    for method_name, model in tqdm(methods.items(), desc="Testing methods"):
        print(f"\n📊 Testing {method_name}...")

        # Train model
        model.fit(user_item_matrix)

        # Test with a few users
        test_users = user_item_matrix.index[:5]

        print(f"   Testing recommendations for {len(test_users)} users...")
        for user_id in test_users[:2]:  # Show first 2 users
            recommendations = model.get_recommendations(user_id, n_recommendations=5)

            if recommendations:
                movie_details = movies[movies["movieId"].isin(recommendations)]
                print(f"   User {user_id} recommendations:")
                for i, (_, movie) in enumerate(movie_details.iterrows()):
                    print(f"     {i+1}. {movie['title']} - {movie['genres']}")
            else:
                print(f"   User {user_id}: No recommendations")

        results[method_name] = model

    return results


if __name__ == "__main__":
    # Load data
    print("Loading MovieLens data...")
    user_item_matrix, movies, ratings = load_and_prepare_data()

    # Compare neighborhood methods
    results = compare_neighborhood_methods(user_item_matrix, movies, ratings)

    print(f"\n✅ Neighborhood methods comparison completed!")
    print(f" Key advantages:")
    print(f"   - Speed: KNN training in seconds vs minutes for MF")
    print(f"   - Interpretability: Clear similarity-based reasoning")
    print(f"   - Scalability: Can handle large datasets efficiently")

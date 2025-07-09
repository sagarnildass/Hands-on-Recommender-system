import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")


def load_movielens_data():
    """
    Load MovieLens data for collaborative filtering
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

    return ratings, movies


def create_user_item_matrix(ratings):
    """
    Create user-item rating matrix for collaborative filtering
    """
    print("Creating user-item rating matrix...")

    # Create pivot table: users as rows, movies as columns, ratings as values
    user_item_matrix = ratings.pivot_table(
        index="userId",
        columns="movieId",
        values="rating",
        fill_value=np.nan,  # Fill missing values with NaN
    )

    print(f"User-item matrix shape: {user_item_matrix.shape}")
    print(f"Number of ratings: {user_item_matrix.notna().sum().sum()}")
    print(
        f"Sparsity: {(1 - user_item_matrix.notna().sum().sum() / (user_item_matrix.shape[0] * user_item_matrix.shape[1])) * 100:.2f}%"
    )

    return user_item_matrix


def calculate_user_similarity(user_item_matrix, user1_id, user2_id, method="cosine"):
    """
    Calculate similarity between two users
    """
    # Get ratings for both users
    user1_ratings = user_item_matrix.loc[user1_id]
    user2_ratings = user_item_matrix.loc[user2_id]

    # Find common movies (where both users have ratings)
    common_movies = user1_ratings.notna() & user2_ratings.notna()

    if common_movies.sum() < 5:  # Need at least 5 common movies
        return 0.0

    # Get ratings for common movies
    user1_common = user1_ratings[common_movies]
    user2_common = user2_ratings[common_movies]

    if method == "cosine":
        # Cosine similarity
        return 1 - cosine(user1_common, user2_common)

    elif method == "pearson":
        # Pearson correlation
        correlation, _ = pearsonr(user1_common, user2_common)
        return correlation if not np.isnan(correlation) else 0.0

    elif method == "jaccard":
        # Jaccard similarity (binary: rated/not rated)
        user1_rated = user1_ratings.notna()
        user2_rated = user2_ratings.notna()
        intersection = (user1_rated & user2_rated).sum()
        union = (user1_rated | user2_rated).sum()
        return intersection / union if union > 0 else 0.0

    else:
        raise ValueError("Method must be 'cosine', 'pearson', or 'jaccard'")


def find_similar_users(user_item_matrix, target_user_id, n_similar=10, method="cosine"):
    """
    Find users similar to the target user
    """
    print(
        f"Finding users similar to User {target_user_id} using {method} similarity..."
    )

    similarities = []

    # Calculate similarity with all other users
    for user_id in user_item_matrix.index:
        if user_id != target_user_id:
            similarity = calculate_user_similarity(
                user_item_matrix, target_user_id, user_id, method
            )
            similarities.append((user_id, similarity))

    # Sort by similarity and return top N
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:n_similar]


def get_user_based_recommendations(
    user_item_matrix, target_user_id, movies, n_recommendations=10, method="cosine"
):
    """
    Improved user-based collaborative filtering with better handling
    """
    # Find similar users
    similar_users = find_similar_users(
        user_item_matrix, target_user_id, n_similar=50, method=method
    )

    # Get target user's rated movies
    target_user_ratings = user_item_matrix.loc[target_user_id]
    rated_movies = target_user_ratings[target_user_ratings.notna()].index

    # Calculate weighted average ratings from similar users
    recommendations = {}

    for movie_id in user_item_matrix.columns:
        if movie_id not in rated_movies:  # Only recommend unrated movies
            weighted_sum = 0
            similarity_sum = 0
            count = 0

            for similar_user_id, similarity in similar_users:
                if similarity > 0.1:  # Higher threshold for similarity
                    rating = user_item_matrix.loc[similar_user_id, movie_id]
                    if pd.notna(rating):
                        weighted_sum += similarity * rating
                        similarity_sum += similarity
                        count += 1

            # Only recommend if we have enough similar users who rated this movie
            if similarity_sum > 0 and count >= 3:
                predicted_rating = weighted_sum / similarity_sum
                recommendations[movie_id] = predicted_rating

    # Sort by predicted rating and get top recommendations
    sorted_recommendations = sorted(
        recommendations.items(), key=lambda x: x[1], reverse=True
    )

    # Get movie details
    top_recommendations = []
    for movie_id, predicted_rating in sorted_recommendations[:n_recommendations]:
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


def test_collaborative_filtering(user_item_matrix, movies, ratings):
    """
    Test collaborative filtering with different similarity methods
    """
    print("\n" + "=" * 60)
    print("👥 TESTING COLLABORATIVE FILTERING")
    print("=" * 60)

    # Find active users for testing
    user_activity = user_item_matrix.notna().sum(axis=1)
    active_users = user_activity.nlargest(5).index

    # Test with first active user
    test_user = active_users[0]
    print(f"\n🎯 Testing with User {test_user} (has {user_activity[test_user]} ratings)")

    # Test different similarity methods
    methods = ["cosine", "pearson", "jaccard"]

    for method in methods:
        print(f"\n📊 Using {method.upper()} similarity:")

        # Get recommendations
        recommendations = get_user_based_recommendations(
            user_item_matrix, test_user, movies, n_recommendations=5, method=method
        )

        print(f"Top 5 recommendations:")
        for i, rec in recommendations.iterrows():
            print(f"  {i+1}. {rec['title']} (Predicted: {rec['predicted_rating']:.2f})")
            print(f"     Genres: {rec['genres']}")


if __name__ == "__main__":
    # Load data
    print("Loading MovieLens data...")
    ratings, movies = load_movielens_data()

    # Create user-item matrix
    user_item_matrix = create_user_item_matrix(ratings)

    # Test collaborative filtering
    test_collaborative_filtering(user_item_matrix, movies, ratings)

    print(f"\n✅ Collaborative filtering implemented successfully!")

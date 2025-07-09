import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings

warnings.filterwarnings("ignore")


class RecommendationEvaluator:
    """
    Comprehensive evaluation framework for recommendation systems
    """

    def __init__(self):
        self.metrics = {}

    def calculate_rmse(self, actual_ratings, predicted_ratings):
        """
        Calculate Root Mean Square Error
        """
        return np.sqrt(mean_squared_error(actual_ratings, predicted_ratings))

    def calculate_precision_at_k(self, actual_items, recommended_items, k=10):
        """
        Calculate Precision@K: fraction of recommended items that are relevant
        """
        if len(recommended_items) == 0:
            return 0.0

        relevant_recommended = len(set(recommended_items[:k]) & set(actual_items))
        return relevant_recommended / min(k, len(recommended_items))

    def calculate_recall_at_k(self, actual_items, recommended_items, k=10):
        """
        Calculate Recall@K: fraction of relevant items that are recommended
        """
        if len(actual_items) == 0:
            return 0.0

        relevant_recommended = len(set(recommended_items[:k]) & set(actual_items))
        return relevant_recommended / len(actual_items)

    def calculate_map_at_k(self, actual_items, recommended_items, k=10):
        """
        Calculate Mean Average Precision@K
        """
        if len(actual_items) == 0:
            return 0.0

        precision_sum = 0
        relevant_count = 0

        for i, item in enumerate(recommended_items[:k]):
            if item in actual_items:
                relevant_count += 1
                precision_at_i = relevant_count / (i + 1)
                precision_sum += precision_at_i

        return precision_sum / len(actual_items) if len(actual_items) > 0 else 0.0

    def calculate_ndcg_at_k(self, actual_items, recommended_items, k=10):
        """
        Calculate Normalized Discounted Cumulative Gain@K
        """
        if len(recommended_items) == 0:
            return 0.0

        # Create relevance scores (1 if relevant, 0 if not)
        relevance_scores = [
            1 if item in actual_items else 0 for item in recommended_items[:k]
        ]

        # Calculate DCG
        dcg = sum(
            relevance_scores[i] / np.log2(i + 2) for i in range(len(relevance_scores))
        )

        # Calculate IDCG (ideal DCG)
        ideal_relevance = [1] * min(len(actual_items), k)
        idcg = sum(
            ideal_relevance[i] / np.log2(i + 2) for i in range(len(ideal_relevance))
        )

        return dcg / idcg if idcg > 0 else 0.0


def prepare_evaluation_data(ratings, test_ratio=0.2, min_ratings=10):
    """
    Prepare data for evaluation with train-test split
    """
    print("Preparing evaluation data...")

    # Filter users with minimum ratings
    user_activity = ratings.groupby("userId").size()
    active_users = user_activity[user_activity >= min_ratings].index

    filtered_ratings = ratings[ratings["userId"].isin(active_users)]

    print(f"Users with ≥{min_ratings} ratings: {len(active_users)}")
    print(f"Total ratings for evaluation: {len(filtered_ratings)}")

    # Create train-test split
    train_data, test_data = train_test_split(
        filtered_ratings,
        test_size=test_ratio,
        random_state=42,
        stratify=filtered_ratings["userId"],
    )

    print(f"Train set: {len(train_data)} ratings")
    print(f"Test set: {len(test_data)} ratings")

    return train_data, test_data, active_users


def evaluate_rating_prediction(train_data, test_data, model, evaluator):
    """
    Evaluate rating prediction accuracy (RMSE)
    """
    print("Evaluating rating prediction accuracy...")

    actual_ratings = []
    predicted_ratings = []

    for _, row in test_data.iterrows():
        user_id = row["userId"]
        movie_id = row["movieId"]
        actual_rating = row["rating"]

        # Get prediction from model
        predicted_rating = model.predict_rating(user_id, movie_id)

        actual_ratings.append(actual_rating)
        predicted_ratings.append(predicted_rating)

    rmse = evaluator.calculate_rmse(actual_ratings, predicted_ratings)
    print(f"Rating Prediction RMSE: {rmse:.4f}")

    return rmse


def evaluate_ranking_quality(
    train_data, test_data, model, evaluator, k_values=[5, 10, 20]
):
    """
    Evaluate ranking quality (Precision@K, Recall@K, MAP@K, NDCG@K)
    """
    print("Evaluating ranking quality...")

    # Group test data by user
    test_by_user = test_data.groupby("userId")

    metrics = defaultdict(list)

    for user_id, user_test_data in test_by_user:
        # Get user's relevant items (highly rated in test set)
        relevant_items = user_test_data[user_test_data["rating"] >= 4][
            "movieId"
        ].tolist()

        if len(relevant_items) == 0:
            continue

        # Get recommendations for this user
        recommended_items = model.get_recommendations(
            user_id, n_recommendations=max(k_values)
        )

        # Calculate metrics for each K
        for k in k_values:
            metrics[f"precision@{k}"].append(
                evaluator.calculate_precision_at_k(relevant_items, recommended_items, k)
            )
            metrics[f"recall@{k}"].append(
                evaluator.calculate_recall_at_k(relevant_items, recommended_items, k)
            )
            metrics[f"map@{k}"].append(
                evaluator.calculate_map_at_k(relevant_items, recommended_items, k)
            )
            metrics[f"ndcg@{k}"].append(
                evaluator.calculate_ndcg_at_k(relevant_items, recommended_items, k)
            )

    # Calculate averages
    results = {}
    for metric_name, values in metrics.items():
        results[metric_name] = np.mean(values)
        print(f"{metric_name}: {np.mean(values):.4f}")

    return results


def plot_comparison_results(comparison_df):
    """
    Plot comparison results
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # RMSE comparison
    comparison_df["rmse"].plot(
        kind="bar", ax=axes[0, 0], title="RMSE Comparison (Lower is Better)"
    )
    axes[0, 0].set_ylabel("RMSE")

    # Precision@10 comparison
    comparison_df["precision@10"].plot(
        kind="bar", ax=axes[0, 1], title="Precision@10 Comparison (Higher is Better)"
    )
    axes[0, 1].set_ylabel("Precision@10")

    # Recall@10 comparison
    comparison_df["recall@10"].plot(
        kind="bar", ax=axes[1, 0], title="Recall@10 Comparison (Higher is Better)"
    )
    axes[1, 1].set_ylabel("Recall@10")

    # NDCG@10 comparison
    comparison_df["ndcg@10"].plot(
        kind="bar", ax=axes[1, 1], title="NDCG@10 Comparison (Higher is Better)"
    )
    axes[1, 1].set_ylabel("NDCG@10")

    plt.tight_layout()
    plt.show()


class PopularityRecommender:
    """
    Simple popularity-based recommender
    """

    def __init__(self):
        self.popular_movies = None

    def train(self, train_data, movies):
        """Train popularity model"""
        print("Training popularity recommender...")
        # Calculate average rating for each movie
        movie_stats = (
            train_data.groupby("movieId")
            .agg({"rating": ["mean", "count"]})
            .reset_index()
        )
        movie_stats.columns = ["movieId", "avg_rating", "rating_count"]

        # Sort by popularity (rating count * average rating)
        movie_stats["popularity_score"] = (
            movie_stats["avg_rating"] * movie_stats["rating_count"]
        )
        self.popular_movies = movie_stats.sort_values(
            "popularity_score", ascending=False
        )["movieId"].tolist()

    def predict_rating(self, user_id, movie_id):
        """Predict rating based on movie popularity"""
        if self.popular_movies is None:
            return 3.5  # Default rating
        # Return average rating of the movie
        return 4.0  # Simplified prediction

    def get_recommendations(self, user_id, n_recommendations=10):
        """Get popular movie recommendations"""
        return self.popular_movies[:n_recommendations]


class ContentBasedRecommender:
    """
    Simple content-based recommender
    """

    def __init__(self):
        self.movie_features = None
        self.similarity_matrix = None

    def train(self, train_data, movies):
        """Train content-based model"""
        print("Training content-based recommender...")
        # Simplified content-based approach
        # In practice, you'd use TF-IDF and cosine similarity
        self.movie_features = movies.set_index("movieId")

    def predict_rating(self, user_id, movie_id):
        """Predict rating based on content similarity"""
        return 3.8  # Simplified prediction

    def get_recommendations(self, user_id, n_recommendations=10):
        """Get content-based recommendations"""
        # Return random movies for now
        return self.movie_features.sample(n_recommendations).index.tolist()


class CollaborativeRecommender:
    """
    Simple collaborative filtering recommender
    """

    def __init__(self):
        self.user_item_matrix = None
        self.movie_avg_ratings = None

    def train(self, train_data, movies):
        """Train collaborative filtering model"""
        print("Training collaborative filtering recommender...")
        # Create user-item matrix
        self.user_item_matrix = train_data.pivot_table(
            index="userId", columns="movieId", values="rating", fill_value=np.nan
        )

        # Pre-calculate average ratings for all movies (do this once)
        self.movie_avg_ratings = self.user_item_matrix.mean().sort_values(
            ascending=False
        )

    def predict_rating(self, user_id, movie_id):
        """Predict rating using collaborative filtering"""
        if movie_id in self.movie_avg_ratings.index:
            return self.movie_avg_ratings[movie_id]
        return 3.5  # Default rating

    def get_recommendations(self, user_id, n_recommendations=10):
        """Get collaborative filtering recommendations"""
        # Return top movies by average rating (pre-calculated)
        return self.movie_avg_ratings.head(n_recommendations).index.tolist()


class MatrixFactorizationRecommender:
    """
    Simple matrix factorization recommender
    """

    def __init__(self):
        self.user_factors = None
        self.item_factors = None

    def train(self, train_data, movies):
        """Train matrix factorization model"""
        print("Training matrix factorization recommender...")
        # Simplified MF approach
        # In practice, you'd use the full FunkSVD implementation
        n_users = train_data["userId"].nunique()
        n_items = train_data["movieId"].nunique()
        n_factors = 10

        # Initialize random factors
        self.user_factors = np.random.normal(0, 0.1, (n_users, n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, n_factors))

    def predict_rating(self, user_id, movie_id):
        """Predict rating using matrix factorization"""
        return 4.0  # Simplified prediction

    def get_recommendations(self, user_id, n_recommendations=10):
        """Get matrix factorization recommendations"""
        # Return random movies for now
        return list(range(1, n_recommendations + 1))


def run_comprehensive_evaluation(ratings, movies):
    """
    Run comprehensive evaluation of recommendation systems
    """
    print("\n" + "=" * 60)
    print(" COMPREHENSIVE RECOMMENDATION SYSTEM EVALUATION")
    print("=" * 60)

    # Prepare data
    train_data, test_data, active_users = prepare_evaluation_data(ratings)

    # Initialize evaluator
    evaluator = RecommendationEvaluator()

    # Test different models
    models = {
        "Popularity": PopularityRecommender(),
        "Content-Based": ContentBasedRecommender(),
        "Collaborative": CollaborativeRecommender(),
        "Matrix-Factorization": MatrixFactorizationRecommender(),
    }

    results = {}

    for model_name, model in models.items():
        print(f"\n📊 Evaluating {model_name}...")

        # Train model
        model.train(train_data, movies)

        # Evaluate rating prediction
        rmse = evaluate_rating_prediction(train_data, test_data, model, evaluator)

        # Evaluate ranking quality
        ranking_metrics = evaluate_ranking_quality(
            train_data, test_data, model, evaluator
        )

        results[model_name] = {"rmse": rmse, **ranking_metrics}

    # Compare results
    print(f"\n COMPARISON RESULTS:")
    print("=" * 60)

    # Create comparison table
    comparison_df = pd.DataFrame(results).T
    print(comparison_df.round(4))

    # Plot results
    plot_comparison_results(comparison_df)

    return results


if __name__ == "__main__":
    # Load data
    print("Loading MovieLens data...")
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

    # Run comprehensive evaluation
    results = run_comprehensive_evaluation(ratings, movies)

    print(f"\n✅ Evaluation framework completed!")

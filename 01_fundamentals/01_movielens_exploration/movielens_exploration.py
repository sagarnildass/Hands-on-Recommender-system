import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from urllib.request import urlretrieve
import zipfile
import os


def download_movielens_data():
    """
    Download the MovieLens 1M dataset if not already present
    """
    ratings_url = "http://files.grouplens.org/datasets/movielens/ml-1m.zip"
    local_file = "ml-1m.zip"

    # Check if the file already exists
    if not os.path.exists("ml-1m"):
        print("Downloading MovieLens 1M dataset...")
        urlretrieve(ratings_url, local_file)

        with zipfile.ZipFile(local_file, "r") as zip_ref:
            zip_ref.extractall()

        # Remove the zip file
        os.remove(local_file)
        print("MovieLens 1M dataset downloaded and extracted.")

    else:
        print("MovieLens 1M dataset already exists.")


def load_movielens_data():
    """
    Load MovieLens 1M Dataset into pandas DataFrame
    """

    ratings = pd.read_csv(
        "ml-1m/ratings.dat",
        sep="::",
        names=["userId", "movieId", "rating", "timestamp"],
        engine="python",
        encoding="latin-1",
    )
    movies = pd.read_csv(
        "ml-1m/movies.dat",
        sep="::",
        names=["movieId", "title", "genres"],
        engine="python",
        encoding="latin-1",
    )
    users = pd.read_csv(
        "ml-1m/users.dat",
        sep="::",
        names=["userId", "gender", "age", "occupation", "zipcode"],
        engine="python",
        encoding="latin-1",
    )

    return ratings, movies, users


def analyze_ratings_distribution(ratings):
    """
    Analyze the distribution of ratings
    """
    print("\n=== Rating Distribution Analysis ===")

    # Basic statistics
    print(f"Average Rating: {ratings['rating'].mean():.2f}")
    print(f"Rating standard deviation: {ratings['rating'].std():.2f}")
    print(f"Min rating: {ratings['rating'].min()}")
    print(f"Max rating: {ratings['rating'].max()}")

    # Rating distribution
    rating_counts = ratings["rating"].value_counts().sort_index()
    print("\nRating Distribution:")

    for rating, count in rating_counts.items():
        percentage = (count / len(ratings)) * 100
        print(f"  {rating} stars: {count:,} ratings ({percentage:.1f}%)")

    # Plot rating distribution
    plt.figure(figsize=(10, 6))
    ratings["rating"].hist(bins=5, edgecolor="black", alpha=0.7)
    plt.title("Distribution of Movie Ratings")
    plt.xlabel("Rating")
    plt.ylabel("Number of Ratings")
    plt.xticks(range(1, 6))
    plt.grid(True, alpha=0.3)
    plt.show()


def analyze_user_item_activity(ratings):
    """
    Analyze user and item activity patterns
    """
    print("\n=== User and Item Activity Analysis ===")

    # User activity analysis
    user_activity = ratings.groupby("userId").size()
    print(f"\nUser Activity:")
    print(f"Average ratings per user: {user_activity.mean():.1f}")
    print(f"Median ratings per user: {user_activity.median():.1f}")
    print(f"Most active user: {user_activity.max()} ratings")
    print(f"Least active user: {user_activity.min()} ratings")

    # Item activity analysis
    item_activity = ratings.groupby("movieId").size()
    print(f"\nItem Activity:")
    print(f"Average ratings per movie: {item_activity.mean():.1f}")
    print(f"Median ratings per movie: {item_activity.median():.1f}")
    print(f"Most rated movie: {item_activity.max()} ratings")
    print(f"Least rated movie: {item_activity.min()} ratings")

    # Activity distribution plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # User activity distribution
    user_activity.hist(bins=50, ax=ax1, alpha=0.7, edgecolor="black")
    ax1.set_title("User Activity Distribution")
    ax1.set_xlabel("Number of Ratings per User")
    ax1.set_ylabel("Number of Users")
    ax1.set_yscale("log")  # Log scale to see the long tail

    # Item activity distribution
    item_activity.hist(bins=50, ax=ax2, alpha=0.7, edgecolor="black")
    ax2.set_title("Item Activity Distribution")
    ax2.set_xlabel("Number of Ratings per Movie")
    ax2.set_ylabel("Number of Movies")
    ax2.set_yscale("log")  # Log scale to see the long tail

    plt.tight_layout()
    plt.show()

def analyze_sparsity(ratings):
    """
    Analyze the sparsity of the user-item matrix
    """
    print("\n=== Sparsity Analysis ===")

    # Calculate sparsity
    total_possible_ratings = len(ratings['userId'].unique()) * len(ratings['movieId'].unique())
    actual_ratings = len(ratings)
    sparsity = (1 - (actual_ratings / total_possible_ratings)) * 100
    
    print(f"Total possible ratings: {total_possible_ratings:,}")
    print(f"Actual ratings: {actual_ratings:,}")
    print(f"Sparsity: {sparsity:.2f}%")
    print(f"Density: {100-sparsity:.2f}%")
    
    # What does this mean?
    print(f"\nInterpretation:")
    print(f"- Only {100-sparsity:.2f}% of all possible user-movie combinations have ratings")
    print(f"- {sparsity:.2f}% of the user-item matrix is empty!")

def calculate_basic_metrics(ratings):
    """
    Calculate basic evaluation metrics for recommendation systems
    """
    print("\n=== Basic Evaluation Metrics ===")
    
    # Calculate RMSE for rating prediction
    # For now, let's calculate it against the mean rating
    mean_rating = ratings['rating'].mean()
    mse = ((ratings['rating'] - mean_rating) ** 2).mean()
    rmse = np.sqrt(mse)
    
    print(f"Mean rating: {mean_rating:.2f}")
    print(f"RMSE (vs mean): {rmse:.2f}")

    # Rating distribution by user activity
    print(f"\nRating patterns by user activity:")
    active_users = ratings.groupby('userId').size() > 100 # Users with at least 100 ratings
    active_user_ratings = ratings[ratings['userId'].isin(active_users[active_users].index)]
    inactive_user_ratings = ratings[ratings['userId'].isin(active_users[~active_users].index)]
    
    print(f"Active users (>100 ratings): {len(active_user_ratings['userId'].unique())}")
    print(f"Active users avg rating: {active_user_ratings['rating'].mean():.2f}")
    print(f"Inactive users avg rating: {inactive_user_ratings['rating'].mean():.2f}")

def generate_summary_insights(ratings, movies, users):
    """
    Generate summary insights and key takeaways
    """
    print("\n" + "="*60)
    print("🎯 KEY INSIGHTS & CHALLENGES FOR RECOMMENDATION SYSTEMS")
    print("="*60)
    
    print("\n📊 DATA CHARACTERISTICS:")
    print(f"• Dataset: {len(ratings):,} ratings, {len(users)} users, {len(movies)} movies")
    print(f"• Sparsity: 95.53% (major challenge!)")
    print(f"• Rating bias: 57.5% are 4-5 stars")
    
    print("\n🎭 USER BEHAVIOR PATTERNS:")
    print(f"• Long tail: Most users rate few movies, few users rate many")
    print(f"• Activity difference: Active users (3.55 avg) vs Inactive users (3.77 avg)")
    print(f"• Rating inflation: Users tend to rate things they like")
    
    print("\n🚨 MAJOR CHALLENGES:")
    print("1. COLD START: New users/items have no ratings")
    print("2. SPARSITY: 95.53% of user-item matrix is empty")
    print("3. RATING BIAS: Most ratings are positive")
    print("4. POPULARITY BIAS: Popular items get more ratings")
    print("5. DIVERSITY: How to recommend diverse items?")
    
    print("\n🔧 ALGORITHM REQUIREMENTS:")
    print("• Must handle sparse data efficiently")
    print("• Should address cold start problems")
    print("• Need to balance accuracy vs diversity")
    print("• Must be scalable for large datasets")
    
    print("\n📈 NEXT STEPS:")
    print("• Project 2: Content-based filtering")
    print("• Project 3: Collaborative filtering")
    print("• Project 4: Matrix factorization")
    print("• Project 5: Evaluation frameworks")
    

if __name__ == "__main__":
    download_movielens_data()
    ratings, movies, users = load_movielens_data()

    # Print basic information about the dataset
    print("\nDataset Overview:")
    print(f"Ratings: {len(ratings)} ratings")
    print(f"Movies: {len(movies)} movies")
    print(f"Users: {len(users)} users")

    print("\nColumn names in ratings DataFrame:")
    print(ratings.columns.tolist())
    print("\nFirst few rows of ratings:")
    print(ratings.head())

    analyze_ratings_distribution(ratings)
    analyze_user_item_activity(ratings)
    analyze_sparsity(ratings)
    calculate_basic_metrics(ratings)
    generate_summary_insights(ratings, movies, users)

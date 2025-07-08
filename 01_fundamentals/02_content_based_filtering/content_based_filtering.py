import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")


def load_and_prepare_data():
    """
    Load MovieLens data and prepare it for content-based filtering
    """
    # Load the data (using the same approach as Project 1)
    ratings = pd.read_csv('../01_movielens_exploration/ml-1m/ratings.dat', 
                         sep='::', 
                         names=['userId', 'movieId', 'rating', 'timestamp'],
                         engine='python',
                         encoding='latin-1')
    
    movies = pd.read_csv('../01_movielens_exploration/ml-1m/movies.dat',
                        sep='::',
                        names=['movieId', 'title', 'genres'],
                        engine='python',
                        encoding='latin-1')
    
    # Merge ratings with movie information
    movie_ratings = ratings.merge(movies, on='movieId')
    
    # Calculate average rating for each movie
    movie_stats = movie_ratings.groupby('movieId').agg({
        'rating': ['mean', 'count'],
        'title': 'first',
        'genres': 'first'
    }).reset_index()
    
    # Flatten column names
    movie_stats.columns = ['movieId', 'avg_rating', 'rating_count', 'title', 'genres']
    
    return movie_stats, movie_ratings


def create_movie_features(movie_stats):
    """
    Create features for content-based filtering
    """
    print("Creating movie features for content-based filtering...")
    
    # 1. TF-IDF for genres (text feature)
    print("Processing genres with TF-IDF...")
    tfidf = TfidfVectorizer(stop_words='english')
    genre_matrix = tfidf.fit_transform(movie_stats['genres'].fillna(''))
    
    # Convert to DataFrame for easier handling
    genre_features = pd.DataFrame(
        genre_matrix.toarray(),
        columns=tfidf.get_feature_names_out(),
        index=movie_stats.index
    )
    
    # 2. Numerical features
    print("Processing numerical features...")
    numerical_features = movie_stats[['avg_rating', 'rating_count']].copy()
    
    # Normalize numerical features
    scaler = StandardScaler()
    numerical_features_scaled = pd.DataFrame(
        scaler.fit_transform(numerical_features),
        columns=['avg_rating_scaled', 'rating_count_scaled'],
        index=movie_stats.index
    )
    
    # 3. Combine all features
    all_features = pd.concat([genre_features, numerical_features_scaled], axis=1)
    
    print(f"Created {all_features.shape[1]} features:")
    print(f"- {genre_features.shape[1]} genre features")
    print(f"- {numerical_features_scaled.shape[1]} numerical features")
    
    return all_features, tfidf, scaler

def build_content_based_recommender(movie_features, movie_stats):
    """
    Build content-based recommendation system
    """
    print("Building content-based recommender...")
    
    # Calculate cosine similarity between all movies
    print("Calculating movie similarities...")
    similarity_matrix = cosine_similarity(movie_features)
    
    # Convert to DataFrame for easier access
    similarity_df = pd.DataFrame(
        similarity_matrix,
        index=movie_stats['movieId'],
        columns=movie_stats['movieId']
    )
    
    print(f"Similarity matrix shape: {similarity_df.shape}")
    print(f"Average similarity: {similarity_matrix.mean():.3f}")
    
    return similarity_df


def get_content_based_recommendations(movie_id, similarity_df, movie_stats, n_recommendations=10):
    """
    Get content-based recommendations for a given movie
    """
    # Get similarity scores for the movie
    movie_similarities = similarity_df.loc[movie_id].sort_values(ascending=False)
    
    # Get top similar movies (excluding the movie itself)
    similar_movies = movie_similarities[1:n_recommendations+1]
    
    # Get movie details
    recommendations = []
    for similar_movie_id, similarity_score in similar_movies.items():
        movie_info = movie_stats[movie_stats['movieId'] == similar_movie_id].iloc[0]
        recommendations.append({
            'movieId': similar_movie_id,
            'title': movie_info['title'],
            'genres': movie_info['genres'],
            'avg_rating': movie_info['avg_rating'],
            'rating_count': movie_info['rating_count'],
            'similarity': similarity_score
        })
    
    return pd.DataFrame(recommendations)


def get_user_recommendations(user_id, movie_ratings, similarity_df, movie_stats, n_recommendations=10):
    """
    Get content-based recommendations for a specific user
    """
    # Get user's rated movies
    user_ratings = movie_ratings[movie_ratings['userId'] == user_id]
    
    if len(user_ratings) == 0:
        print(f"User {user_id} has no ratings.")
        return None
    
    # Get user's liked movies (rating >= 4)
    liked_movies = user_ratings[user_ratings['rating'] >= 4]['movieId'].tolist()
    
    if len(liked_movies) == 0:
        print(f"User {user_id} has no highly rated movies.")
        return None
    
    # Calculate average similarity to user's liked movies
    user_similarities = similarity_df.loc[liked_movies].mean()
    
    # Remove movies the user has already rated
    rated_movies = user_ratings['movieId'].tolist()
    user_similarities = user_similarities.drop(rated_movies)
    
    # Get top recommendations
    top_recommendations = user_similarities.nlargest(n_recommendations)
    
    # Get movie details
    recommendations = []
    for movie_id, similarity_score in top_recommendations.items():
        movie_info = movie_stats[movie_stats['movieId'] == movie_id].iloc[0]
        recommendations.append({
            'movieId': movie_id,
            'title': movie_info['title'],
            'genres': movie_info['genres'],
            'avg_rating': movie_info['avg_rating'],
            'rating_count': movie_info['rating_count'],
            'similarity': similarity_score
        })
    
    return pd.DataFrame(recommendations)


def test_content_based_recommender(similarity_df, movie_stats, movie_ratings):
    """
    Test the content-based recommender with some example movies and users
    """
    print("\n" + "="*60)
    print("🎬 TESTING CONTENT-BASED RECOMMENDER")
    print("="*60)
    
    # Find some popular movies to test with
    popular_movies = movie_stats.nlargest(10, 'rating_count')[['movieId', 'title', 'genres']]
    
    print("\nTesting with popular movies:")
    for idx, movie in popular_movies.head(3).iterrows():
        print(f"\n🎯 Original Movie: {movie['title']}")
        print(f"   Genres: {movie['genres']}")
        
        # Get recommendations
        recommendations = get_content_based_recommendations(
            movie['movieId'], similarity_df, movie_stats, n_recommendations=5
        )
        
        print(f"\n📋 Top 5 Recommendations:")
        for i, rec in recommendations.iterrows():
            print(f"   {i+1}. {rec['title']} (Similarity: {rec['similarity']:.3f})")
            print(f"      Genres: {rec['genres']}")
            print(f"      Avg Rating: {rec['avg_rating']:.2f} ({rec['rating_count']} ratings)")
    
    # Test with different genres
    print(f"\n🎭 Testing genre diversity...")
    genre_examples = {
        'Action': movie_stats[movie_stats['genres'].str.contains('Action', na=False)].iloc[0],
        'Comedy': movie_stats[movie_stats['genres'].str.contains('Comedy', na=False)].iloc[0],
        'Drama': movie_stats[movie_stats['genres'].str.contains('Drama', na=False)].iloc[0]
    }
    
    for genre, movie in genre_examples.items():
        print(f"\n🎯 {genre} Movie: {movie['title']}")
        recommendations = get_content_based_recommendations(
            movie['movieId'], similarity_df, movie_stats, n_recommendations=3
        )
        print(f"   Recommendations: {', '.join(recommendations['title'].tolist())}")

    # Test user-based recommendations
    print(f"\n👥 Testing user-based recommendations...")
    
    # Find users with many ratings
    active_users = movie_ratings.groupby('userId').size().nlargest(5)
    
    for user_id in active_users.index[:2]:  # Test first 2 active users
        print(f"\n🎯 User {user_id} (has {active_users[user_id]} ratings):")
        
        # Get user's top rated movies
        user_top_movies = movie_ratings[movie_ratings['userId'] == user_id].nlargest(3, 'rating')
        print("   Top rated movies:")
        for _, movie in user_top_movies.iterrows():
            movie_info = movie_stats[movie_stats['movieId'] == movie['movieId']].iloc[0]
            print(f"   - {movie_info['title']} ({movie['rating']} stars) - {movie_info['genres']}")
        
        # Get recommendations
        recommendations = get_user_recommendations(user_id, movie_ratings, similarity_df, movie_stats, n_recommendations=5)
        
        if recommendations is not None:
            print(f"\n📋 Top 5 Recommendations:")
            for i, rec in recommendations.iterrows():
                print(f"   {i+1}. {rec['title']} (Similarity: {rec['similarity']:.3f})")
                print(f"      Genres: {rec['genres']}")
        else:
            print("   No recommendations available.")


if __name__ == "__main__":
    # Load and prepare data
    print("Loading MovieLens data...")
    movie_stats, movie_ratings = load_and_prepare_data()
    
    # Create features
    movie_features, tfidf, scaler = create_movie_features(movie_stats)
    
    # Build recommender
    similarity_df = build_content_based_recommender(movie_features, movie_stats)
    
    # Test the recommender
    test_content_based_recommender(similarity_df, movie_stats, movie_ratings)
    
    print(f"\n✅ Content-based recommender built successfully!")
    print(f"📊 Dataset: {len(movie_stats)} movies with {movie_features.shape[1]} features")


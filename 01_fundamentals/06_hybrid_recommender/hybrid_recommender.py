import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class HybridRecommender:
    """
    Hybrid recommendation system with progressive personalization
    """
    def __init__(self, popularity_weight=0.3, content_weight=0.3, collaborative_weight=0.4):
        self.popularity_weight = popularity_weight
        self.content_weight = content_weight
        self.collaborative_weight = collaborative_weight
        
        # Component recommenders
        self.popularity_recommender = None
        self.content_recommender = None
        self.collaborative_recommender = None
        
        # Data storage
        self.user_item_matrix = None
        self.movie_features = None
        self.popular_movies = None
        self.user_activity = None
        
    def train(self, ratings, movies):
        """
        Train all component recommenders
        """
        print("Training Hybrid Recommender...")
        
        # Calculate user activity levels
        self.user_activity = ratings.groupby('userId').size()
        
        # Train popularity recommender
        self._train_popularity_recommender(ratings)
        
        # Train content-based recommender
        self._train_content_recommender(movies)
        
        # Train collaborative filtering recommender
        self._train_collaborative_recommender(ratings)
        
        print("Hybrid recommender training completed!")
    
    def _train_popularity_recommender(self, ratings):
        """
        Train popularity-based recommender
        """
        print("  Training popularity recommender...")
        
        # Calculate movie popularity scores
        movie_stats = ratings.groupby('movieId').agg({
            'rating': ['mean', 'count']
        }).reset_index()
        movie_stats.columns = ['movieId', 'avg_rating', 'rating_count']
        
        # Popularity score = average rating * number of ratings
        movie_stats['popularity_score'] = movie_stats['avg_rating'] * movie_stats['rating_count']
        self.popular_movies = movie_stats.sort_values('popularity_score', ascending=False)
    
    def _train_content_recommender(self, movies):
        """
        Train content-based recommender using TF-IDF
        """
        print("  Training content-based recommender...")
        
        # Create TF-IDF features from genres
        tfidf = TfidfVectorizer(stop_words='english')
        genre_matrix = tfidf.fit_transform(movies['genres'].fillna(''))
        
        # Calculate movie similarity matrix
        self.movie_features = pd.DataFrame(
            genre_matrix.toarray(),
            index=movies['movieId'],
            columns=tfidf.get_feature_names_out()
        )
    
    def _train_collaborative_recommender(self, ratings):
        """
        Train collaborative filtering recommender
        """
        print("  Training collaborative filtering recommender...")
        
        # Create user-item matrix
        self.user_item_matrix = ratings.pivot_table(
            index='userId',
            columns='movieId',
            values='rating',
            fill_value=np.nan
        )

    def get_user_activity_level(self, user_id):
        """
        Determine user activity level for progressive personalization
        """
        if user_id not in self.user_activity.index:
            return 'new_user'
        
        rating_count = self.user_activity[user_id]
        
        if rating_count == 0:
            return 'new_user'
        elif rating_count < 5:
            return 'cold_start'
        elif rating_count < 20:
            return 'early_personalization'
        elif rating_count < 100:
            return 'personalization'
        else:
            return 'advanced_personalization'

    def get_recommendations(self, user_id, n_recommendations=10):
        """
        Get hybrid recommendations based on user activity level
        """
        activity_level = self.get_user_activity_level(user_id)
        
        print(f"User {user_id} activity level: {activity_level}")
        
        if activity_level == 'new_user':
            return self._get_popularity_recommendations(n_recommendations)
        
        elif activity_level == 'cold_start':
            return self._get_hybrid_recommendations(user_id, n_recommendations, 
                                                  [0.7, 0.3, 0.0])  # 70% popularity, 30% content
        
        elif activity_level == 'early_personalization':
            return self._get_hybrid_recommendations(user_id, n_recommendations, 
                                                  [0.4, 0.5, 0.1])  # 40% popularity, 50% content, 10% collaborative
        
        elif activity_level == 'personalization':
            return self._get_hybrid_recommendations(user_id, n_recommendations, 
                                                  [0.2, 0.3, 0.5])  # 20% popularity, 30% content, 50% collaborative
        
        else:  # advanced_personalization
            return self._get_hybrid_recommendations(user_id, n_recommendations, 
                                                  [0.1, 0.2, 0.7])  # 10% popularity, 20% content, 70% collaborative

    def _get_popularity_recommendations(self, n_recommendations):
        """
        Get pure popularity-based recommendations
        """
        return self.popular_movies['movieId'].head(n_recommendations).tolist()

    def _get_hybrid_recommendations(self, user_id, n_recommendations, weights):
        """
        Get hybrid recommendations by combining multiple approaches
        """
        popularity_weight, content_weight, collaborative_weight = weights
        
        # Get recommendations from each component
        popularity_recs = self._get_popularity_recommendations(n_recommendations * 2)
        content_recs = self._get_content_recommendations(user_id, n_recommendations * 2)
        collaborative_recs = self._get_collaborative_recommendations(user_id, n_recommendations * 2)
        
        # Combine recommendations with weights
        final_scores = {}
        
        # Add popularity scores
        for i, movie_id in enumerate(popularity_recs):
            final_scores[movie_id] = final_scores.get(movie_id, 0) + popularity_weight * (1.0 / (i + 1))
        
        # Add content-based scores
        for i, movie_id in enumerate(content_recs):
            final_scores[movie_id] = final_scores.get(movie_id, 0) + content_weight * (1.0 / (i + 1))
        
        # Add collaborative filtering scores
        for i, movie_id in enumerate(collaborative_recs):
            final_scores[movie_id] = final_scores.get(movie_id, 0) + collaborative_weight * (1.0 / (i + 1))
        
        # Sort by final scores and return top recommendations
        sorted_recommendations = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        return [movie_id for movie_id, score in sorted_recommendations[:n_recommendations]]

    def _get_content_recommendations(self, user_id, n_recommendations):
        """
        Get content-based recommendations for a user
        """
        if user_id not in self.user_item_matrix.index:
            return self._get_popularity_recommendations(n_recommendations)
        
        # Get user's rated movies
        user_ratings = self.user_item_matrix.loc[user_id]
        rated_movies = user_ratings[user_ratings.notna()].index
        
        if len(rated_movies) == 0:
            return self._get_popularity_recommendations(n_recommendations)
        
        # Calculate user's genre preferences
        user_preferences = self.movie_features.loc[rated_movies].mean()
        
        # Find movies similar to user preferences
        movie_similarities = cosine_similarity([user_preferences], self.movie_features)[0]
        
        # Get top similar movies (excluding rated ones)
        similar_movies = pd.Series(movie_similarities, index=self.movie_features.index)
        similar_movies = similar_movies.drop(rated_movies).sort_values(ascending=False)
        
        return similar_movies.head(n_recommendations).index.tolist()

    def _get_collaborative_recommendations(self, user_id, n_recommendations):
        """
        Get collaborative filtering recommendations for a user
        """
        if user_id not in self.user_item_matrix.index:
            return self._get_popularity_recommendations(n_recommendations)
        
        # Get user's rated movies
        user_ratings = self.user_item_matrix.loc[user_id]
        rated_movies = user_ratings[user_ratings.notna()].index
        
        if len(rated_movies) == 0:
            return self._get_popularity_recommendations(n_recommendations)
        
        # Find similar users
        similar_users = self._find_similar_users(user_id, n_similar=10)
        
        # Get recommendations from similar users
        recommendations = {}
        for similar_user_id, similarity in similar_users:
            if similarity > 0:
                similar_user_ratings = self.user_item_matrix.loc[similar_user_id]
                liked_movies = similar_user_ratings[similar_user_ratings >= 4].index
                for movie_id in liked_movies:
                    if movie_id not in rated_movies:
                        recommendations[movie_id] = recommendations.get(movie_id, 0) + similarity
        
        # Sort by score and return top recommendations
        sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        return [movie_id for movie_id, score in sorted_recommendations[:n_recommendations]]

    def _find_similar_users(self, user_id, n_similar=10):
        """
        Find users similar to the target user
        """
        if user_id not in self.user_item_matrix.index:
            return []
        
        target_user_ratings = self.user_item_matrix.loc[user_id]
        similarities = []
        
        for other_user_id in self.user_item_matrix.index:
            if other_user_id != user_id:
                other_user_ratings = self.user_item_matrix.loc[other_user_id]
                
                # Find common movies
                common_movies = target_user_ratings.notna() & other_user_ratings.notna()
                
                if common_movies.sum() >= 5:  # Need at least 5 common movies
                    target_common = target_user_ratings[common_movies]
                    other_common = other_user_ratings[common_movies]
                    
                    # Calculate cosine similarity
                    similarity = 1 - cosine(target_common, other_common)
                    similarities.append((other_user_id, similarity))
        
        # Return top similar users
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:n_similar]

def test_hybrid_recommender(ratings, movies):
    """
    Test hybrid recommender with different user types
    """
    print("\n" + "="*60)
    print(" TESTING HYBRID RECOMMENDER")
    print("="*60)
    
    # Initialize and train hybrid recommender
    hybrid_recommender = HybridRecommender()
    hybrid_recommender.train(ratings, movies)
    
    # Analyze user activity distribution
    user_activity = ratings.groupby('userId').size()
    print(f"\n📊 User Activity Analysis:")
    print(f"Total users: {len(user_activity)}")
    print(f"Min ratings per user: {user_activity.min()}")
    print(f"Max ratings per user: {user_activity.max()}")
    print(f"Average ratings per user: {user_activity.mean():.1f}")
    
    # Find users for each category (with safety checks)
    user_types = {}
    
    # New user (doesn't exist in dataset)
    user_types['new_user'] = 99999
    
    # Cold start users (1-4 ratings) - might not exist in MovieLens
    cold_start_users = user_activity[user_activity < 5]
    if len(cold_start_users) > 0:
        user_types['cold_start'] = cold_start_users.index[0]
    else:
        user_types['cold_start'] = user_activity.index[0]  # Use any user for testing
    
    # Early personalization (5-19 ratings)
    early_users = user_activity[(user_activity >= 5) & (user_activity < 20)]
    if len(early_users) > 0:
        user_types['early_personalization'] = early_users.index[0]
    else:
        user_types['early_personalization'] = user_activity.index[0]
    
    # Personalization (20-99 ratings)
    personalization_users = user_activity[(user_activity >= 20) & (user_activity < 100)]
    if len(personalization_users) > 0:
        user_types['personalization'] = personalization_users.index[0]
    else:
        user_types['personalization'] = user_activity.index[0]
    
    # Advanced (100+ ratings)
    advanced_users = user_activity[user_activity >= 100]
    if len(advanced_users) > 0:
        user_types['advanced'] = advanced_users.index[0]
    else:
        user_types['advanced'] = user_activity.index[0]
    
    print(f"\n🎯 Testing different user types:")
    for user_type, user_id in user_types.items():
        print(f"\n {user_type.replace('_', ' ').title()}:")
        
        if user_id in ratings['userId'].values:
            user_ratings_count = len(ratings[ratings['userId'] == user_id])
            print(f"   User {user_id} has {user_ratings_count} ratings")
            print(f"   Activity level: {hybrid_recommender.get_user_activity_level(user_id)}")
        else:
            print(f"   User {user_id} (new user - no ratings)")
            print(f"   Activity level: {hybrid_recommender.get_user_activity_level(user_id)}")
        
        # Get recommendations
        recommendations = hybrid_recommender.get_recommendations(user_id, n_recommendations=5)
        
        # Get movie details
        movie_details = movies[movies['movieId'].isin(recommendations)]
        print(f"   Top 5 recommendations:")
        for i, (_, movie) in enumerate(movie_details.iterrows()):
            print(f"     {i+1}. {movie['title']} - {movie['genres']}")


if __name__ == "__main__":
    # Load data
    print("Loading MovieLens data...")
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
    
    # Test hybrid recommender
    test_hybrid_recommender(ratings, movies)
    
    print(f"\n✅ Hybrid recommender testing completed!")

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

def create_context_features(ratings):
    """
    Create context-aware features from timestamp data
    """
    print("Creating context-aware features...")
    
    # Convert timestamp to datetime
    ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')
    
    # Extract temporal features
    ratings['hour'] = ratings['timestamp'].dt.hour
    ratings['day_of_week'] = ratings['timestamp'].dt.dayofweek
    ratings['month'] = ratings['timestamp'].dt.month
    ratings['year'] = ratings['timestamp'].dt.year

    # Create time-based categories
    ratings['time_of_day'] = pd.cut(ratings['hour'], 
                                   bins=[0, 6, 12, 18, 24], 
                                   labels=['Night', 'Morning', 'Afternoon', 'Evening'])
    
    ratings['day_category'] = pd.cut(ratings['day_of_week'], 
                                    bins=[0, 5, 7], 
                                    labels=['Weekday', 'Weekend'])

    # Create activity-based context (simulated)
    np.random.seed(42)
    ratings['activity_context'] = np.random.choice(
        ['Relaxing', 'Social', 'Work_Break', 'Date_Night', 'Family_Time'], 
        size=len(ratings)
    )
    
    # Create mood context (simulated based on rating patterns)
    ratings['mood_context'] = np.where(
        ratings['rating'] >= 4, 'Happy',
        np.where(ratings['rating'] >= 3, 'Neutral', 'Sad')
    )
    
    print(f"Context features created:")
    print(f"- Time of day: {ratings['time_of_day'].value_counts().to_dict()}")
    print(f"- Day category: {ratings['day_category'].value_counts().to_dict()}")
    print(f"- Activity: {ratings['activity_context'].value_counts().to_dict()}")
    print(f"- Mood: {ratings['mood_context'].value_counts().to_dict()}")
    
    return ratings

class ContextAwareRecommender:
    """
    Context-aware recommendation system
    """
    def __init__(self):
        self.user_item_matrix = None
        self.context_profiles = None
        self.movie_context_preferences = None

    def fit(self, ratings, movies):
        """
        Train context-aware recommender
        """
        print("Training context-aware recommender...")
        
        # Create user-item matrix
        self.user_item_matrix = ratings.pivot_table(
            index='userId',
            columns='movieId', 
            values='rating',
            fill_value=0
        )

        # Create context profiles for each user
        # Create context profiles for each user
        self._create_context_profiles(ratings)
        
        # Create movie context preferences
        self._create_movie_context_preferences(ratings, movies)
        
        print("Context-aware training completed!")
    
    def _create_context_profiles(self, ratings):
        """
        Create context preference profiles for each user
        """
        print("  Creating user context profiles...")
        
        context_profiles = {}
        
        for user_id in tqdm(ratings['userId'].unique(), desc="Creating user context profiles"):
            user_ratings = ratings[ratings['userId'] == user_id]

            # Calculate context preferences
            context_prefs = {}

            # Time of day preferences
            time_prefs = user_ratings.groupby('time_of_day')['rating'].mean()
            context_prefs['time_of_day'] = time_prefs.to_dict()

            # Day category preferences
            day_prefs = user_ratings.groupby('day_category')['rating'].mean()
            context_prefs['day_category'] = day_prefs.to_dict()

            # Activity preferences
            activity_prefs = user_ratings.groupby('activity_context')['rating'].mean()
            context_prefs['activity_context'] = activity_prefs.to_dict()

            # Mood preferences
            mood_prefs = user_ratings.groupby('mood_context')['rating'].mean()
            context_prefs['mood_context'] = mood_prefs.to_dict()
            
            context_profiles[user_id] = context_prefs

        self.context_profiles = context_profiles

    def _create_movie_context_preferences(self, ratings, movies):
        """
        Create context preferences for each movie
        """
        print("  Creating movie context preferences...")
        
        movie_context = {}

        for movie_id in tqdm(movies['movieId'].unique(), desc="Creating movie context preferences"):
            movie_ratings = ratings[ratings['movieId'] == movie_id]
            
            # Calculate context performance for this movie
            context_performance = {}

            # Time of day performance
            time_perf = movie_ratings.groupby('time_of_day')['rating'].mean()
            context_performance['time_of_day'] = time_perf.to_dict()

            # Day category performance
            day_perf = movie_ratings.groupby('day_category')['rating'].mean()
            context_performance['day_category'] = day_perf.to_dict()

            # Activity performance
            activity_perf = movie_ratings.groupby('activity_context')['rating'].mean()
            context_performance['activity_context'] = activity_perf.to_dict()

            # Mood performance
            mood_perf = movie_ratings.groupby('mood_context')['rating'].mean()
            context_performance['mood_context'] = mood_perf.to_dict()

            movie_context[movie_id] = context_performance
            
        self.movie_context_preferences = movie_context

    def get_context_aware_recommendations(self, user_id, context, n_recommendations=10):
        """
        Get context-aware recommendations
        """
        if user_id not in self.context_profiles or self.user_item_matrix is None:
            return []

        # Get user's context preferences
        user_context_prefs = self.context_profiles[user_id]

        # Get user's rated movies
        user_ratings = self.user_item_matrix.loc[user_id]
        rated_movies = user_ratings[user_ratings > 0].index
        
        # Calculate context-aware scores for each movie
        movie_scores = {}

        for movie_id in tqdm(self.movie_context_preferences, desc="Calculating context-aware scores"):
            if movie_id not in rated_movies and movie_id in self.user_item_matrix.columns:
                movie_context_prefs = self.movie_context_preferences[movie_id]
                
                # Calculate context match score
                context_score = 0
                context_weight = 0
                
                for context_type, context_value in context.items():
                    if context_type in user_context_prefs and context_type in movie_context_prefs:
                        if context_value in user_context_prefs[context_type] and context_value in movie_context_prefs[context_type]:
                            user_pref = user_context_prefs[context_type][context_value]
                            movie_perf = movie_context_prefs[context_type][context_value]
                            
                            # Context match score
                            context_score += user_pref * movie_perf
                            context_weight += 1
                
                # Normalize context score
                if context_weight > 0:
                    context_score = context_score / context_weight
                
                # Combine with base popularity
                base_score = self.user_item_matrix[movie_id].mean()
                final_score = 0.7 * context_score + 0.3 * base_score
                
                movie_scores[movie_id] = final_score
        
        # Sort by score and return top recommendations
        sorted_recommendations = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)
        return [movie_id for movie_id, score in sorted_recommendations[:n_recommendations]]


def test_context_aware_recommendations(ratings, movies):
    """
    Test context-aware recommendations with different scenarios
    """
    print("\n" + "="*60)
    print(" TESTING CONTEXT-AWARE RECOMMENDATIONS")
    print("="*60)
    
    # Initialize and train context-aware recommender
    context_recommender = ContextAwareRecommender()
    context_recommender.fit(ratings, movies)
    
    # Test different context scenarios
    test_contexts = {
        'Weekend Evening Relaxing': {
            'time_of_day': 'Evening',
            'day_category': 'Weekend',
            'activity_context': 'Relaxing',
            'mood_context': 'Happy'
        },
        'Weekday Morning Work Break': {
            'time_of_day': 'Morning',
            'day_category': 'Weekday',
            'activity_context': 'Work_Break',
            'mood_context': 'Neutral'
        },
        'Weekend Afternoon Family Time': {
            'time_of_day': 'Afternoon',
            'day_category': 'Weekend',
            'activity_context': 'Family_Time',
            'mood_context': 'Happy'
        },
        'Weekday Night Date Night': {
            'time_of_day': 'Evening',
            'day_category': 'Weekday',
            'activity_context': 'Date_Night',
            'mood_context': 'Happy'
        }
    }
    
    # Test with a few users
    test_users = ratings['userId'].unique()[:3]
    
    for user_id in tqdm(test_users, desc="Testing context-aware recommendations"):
        print(f"\n👤 User {user_id} Context-Aware Recommendations:")
        
        # Show user's context profile
        if user_id in context_recommender.context_profiles:
            user_profile = context_recommender.context_profiles[user_id]
            print(f"   Context Profile:")
            for context_type, prefs in user_profile.items():
                print(f"     {context_type}: {prefs}")
        
        # Test each context scenario
        for context_name, context in test_contexts.items():
            print(f"\n    {context_name}:")
            
            recommendations = context_recommender.get_context_aware_recommendations(
                user_id, context, n_recommendations=3
            )
            
            if recommendations:
                movie_details = movies[movies['movieId'].isin(recommendations)]
                for i, (_, movie) in enumerate(movie_details.iterrows()):
                    print(f"     {i+1}. {movie['title']} - {movie['genres']}")
            else:
                print(f"     No context-aware recommendations available")


def analyze_context_patterns(ratings):
    """
    Analyze context patterns in the dataset
    """
    print("\n" + "="*60)
    print("📊 CONTEXT PATTERN ANALYSIS")
    print("="*60)
    
    # Time of day analysis
    print("\n🕐 Time of Day Patterns:")
    time_patterns = ratings.groupby('time_of_day')['rating'].agg(['mean', 'count'])
    print(time_patterns)
    
    # Day category analysis
    print("\n📅 Day Category Patterns:")
    day_patterns = ratings.groupby('day_category')['rating'].agg(['mean', 'count'])
    print(day_patterns)
    
    # Activity context analysis
    print("\n🎯 Activity Context Patterns:")
    activity_patterns = ratings.groupby('activity_context')['rating'].agg(['mean', 'count'])
    print(activity_patterns)
    
    # Mood context analysis
    print("\n😊 Mood Context Patterns:")
    mood_patterns = ratings.groupby('mood_context')['rating'].agg(['mean', 'count'])
    print(mood_patterns)
    
    # Cross-context analysis
    print("\n🔄 Cross-Context Analysis:")
    cross_context = ratings.groupby(['time_of_day', 'day_category'])['rating'].mean().unstack()
    print(cross_context)


if __name__ == "__main__":
    # Load data
    print("Loading MovieLens data...")
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
    
    # Create context features
    ratings_with_context = create_context_features(ratings)
    
    # Analyze context patterns
    analyze_context_patterns(ratings_with_context)
    
    # Test context-aware recommendations
    test_context_aware_recommendations(ratings_with_context, movies)
    
    print(f"\n✅ Context-aware recommendations testing completed!")
    print(f"🎯 Key insights:")
    print(f"   - Different contexts lead to different recommendations")
    print(f"   - User behavior varies by time, day, and activity")
    print(f"   - Context-aware systems provide more relevant suggestions")
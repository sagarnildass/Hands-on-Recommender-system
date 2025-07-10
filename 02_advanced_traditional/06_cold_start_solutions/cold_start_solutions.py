import numpy as np
import pandas as pd
import time
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from tqdm import tqdm
import warnings
from typing import Dict, List, Tuple, Optional
warnings.filterwarnings('ignore')


class ColdStartSolutions:
    """
    Solutions for handling cold start problems in recommendation systems.
    Addresses both user cold start and item cold start scenarios.
    """
    def __init__(self, n_clusters: int = 10, min_similarity: float = 0.1):
        self.n_clusters = n_clusters
        self.min_similarity = min_similarity
        self.user_profiles = {}
        self.item_profiles = {}
        self.user_clusters = {}
        self.item_clusters = {}
        self.popularity_scores = {}
        self.content_features = {}
        self.tfidf_vectorizer = None

    def _calculate_popularity_scores(self, ratings: pd.DataFrame) -> Dict[int, float]:
        """
        Calculate popularity scores for items based on rating frequency and average rating
        """
        print("📊 Calculating popularity scores...")

        # Count ratings per item
        item_counts = ratings.groupby('movieId').size()

        # Average rating per item
        item_avg_ratings = ratings.groupby('movieId')['rating'].mean()

        # Combine frequency and rating (weighted popularity)
        popularity_scores = {}

        for item_id in tqdm(item_counts.index, desc="Calculating popularity scores"):
            count = item_counts[item_id]
            avg_rating = item_avg_ratings[item_id]

            # Popularity = frequency * average rating (normalized)
            popularity = (count / item_counts.max()) * avg_rating
            popularity_scores[item_id] = popularity
        
        return popularity_scores

    def _extract_content_features(self, movies: pd.DataFrame) -> Dict[int, str]:
        """
        Extract content features from movie metadata for cold start
        """
        print("🎥 Extracting content features from movies...")

        content_features = {}

        for _, movie in tqdm(movies.iterrows(), desc="Extracting content features"):
            movie_id = movie['movieId']

            # Combine title, genres and year into a single text feature
            title = str(movie.get('title', ''))
            genres = str(movie.get('genres', ''))
            year = str(movie.get('year', ''))

            # Create combined feature string
            feature_text = f"{title} {genres} {year}".lower()
            content_features[movie_id] = feature_text

        return content_features

    def _build_tfidf_features(self, content_features: Dict[int, str]) -> np.ndarray:
        """
        Build TF-IDF features from content text
        """
        print("🔤 Building TF-IDF features...")

        # Prepare text data
        movie_ids = list(content_features.keys())
        texts = [content_features[movie_id] for movie_id in movie_ids]

        # Create TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )

        tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)

        # Store mapping
        self.movie_id_to_idx = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}
        self.idx_to_movie_id = {idx: movie_id for movie_id, idx in self.movie_id_to_idx.items()}
        
        return tfidf_matrix.toarray()

    def _create_user_clusters(self, ratings: pd.DataFrame) -> Dict[int, int]:
        """
        Create user clusters based on rating patterns for cold start
        """
        print("👥 Creating user clusters...")

        # Create user-item matrix
        user_item_matrix = ratings.pivot_table(
            index='userId',
            columns='movieId',
            values='rating',
            fill_value=0
        )

        # Use K-means clustering
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        user_clusters = kmeans.fit_predict(user_item_matrix)

        # Create mapping
        user_cluster_map = {}
        for i, user_id in tqdm(enumerate(user_item_matrix.index), desc="Mapping users to clusters"):
            user_cluster_map[user_id] = user_clusters[i]

        return user_cluster_map
    
    def _create_item_clusters(self, tfidf_features: np.ndarray) -> Dict[int, int]:
        """
        Create item clusters based on content features
        """
        print("🎭 Creating item clusters...")

        # Use K-means clustering on TF-IDF features
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        item_clusters = kmeans.fit_predict(tfidf_features)

        # Create mapping
        item_cluster_map = {}
        for idx, cluster_id in tqdm(enumerate(item_clusters), desc="Mapping items to clusters"):
            movie_id = self.idx_to_movie_id[idx]
            item_cluster_map[movie_id] = cluster_id
        
        return item_cluster_map

    def fit(self, ratings: pd.DataFrame, movies: pd.DataFrame):
        """
        Fit the cold start solution system
        """
        print("🚀 Training cold start solutions...")
        start_time = time.time()

        # Calculate popularity scores
        self.popularity_scores = self._calculate_popularity_scores(ratings)

        # Extract content features
        self.content_features = self._extract_content_features(movies)

        # Build TF-IDF features
        tfidf_features = self._build_tfidf_features(self.content_features)
        
        # Create clusters
        self.user_clusters = self._create_user_clusters(ratings)
        self.item_clusters = self._create_item_clusters(tfidf_features)

        # Store TF-IDF features for similarity calculations
        self.tfidf_features = tfidf_features

        training_time = time.time() - start_time
        print(f"✅ Cold start training completed in {training_time:.2f} seconds")
    
    def recommend_for_new_user(self, user_id: int, n_recommendations: int = 10, 
                              method: str = 'popularity') -> List[Tuple[int, float]]:
        """
        Generate recommendations for a new user (cold start)
        """
        print(f"🆕 Generating recommendations for new user {user_id}...")
        
        if method == 'popularity':
            return self._popularity_recommendations(n_recommendations)
        elif method == 'cluster':
            return self._cluster_based_recommendations(user_id, n_recommendations)
        elif method == 'diversity':
            return self._diversity_recommendations(n_recommendations)
        else:
            raise ValueError(f"Unknown method: {method}")

    def recommend_for_new_item(self, item_id: int, n_recommendations: int = 10) -> List[Tuple[int, float]]:
        """
        Generate recommendations for a new item (cold start)
        """
        print(f"🆕 Generating recommendations for new item {item_id}...")
        
        if item_id in self.content_features:
            return self._content_based_item_recommendations(item_id, n_recommendations)
        else:
            print(f"⚠️  Item {item_id} not found in content features")
            return []

    def _popularity_recommendations(self, n_recommendations: int) -> List[Tuple[int, float]]:
        """
        Popularity-based recommendations for cold start
        """
        # Sort items by popularity score
        sorted_items = sorted(self.popularity_scores.items(), 
                            key=lambda x: x[1], reverse=True)
        
        recommendations = []
        for item_id, score in tqdm(sorted_items[:n_recommendations], desc="Generating popularity recommendations"):
            recommendations.append((item_id, score))
        
        return recommendations

    def _cluster_based_recommendations(self, user_id: int, n_recommendations: int) -> List[Tuple[int, float]]:
        """
        Cluster-based recommendations for new users
        """
        # For new users, we can't determine their cluster directly
        # So we'll use a random cluster and recommend popular items from that cluster
        import random

        # Randomly assign user to a cluster
        user_cluster = random.randint(0, self.n_clusters - 1)

        # Find items in the same cluster
        cluster_items = []
        for item_id, cluster_id in tqdm(self.item_clusters.items(), desc="Finding cluster items"):
            if cluster_id == user_cluster:
                popularity = self.popularity_scores.get(item_id, 0)
                cluster_items.append((item_id, popularity))

        # Sort by popularity
        cluster_items.sort(key=lambda x: x[1], reverse=True)

        return cluster_items[:n_recommendations]

    def _diversity_recommendations(self, n_recommendations: int) -> List[Tuple[int, float]]:
        """
        Diversity-based recommendations to explore different genres
        """
        # Group items by cluster and select top items from each cluster
        cluster_items = defaultdict(list)

        for item_id, cluster_id in tqdm(self.item_clusters.items(), desc="Grouping items by cluster"):
            popularity = self.popularity_scores.get(item_id, 0)
            cluster_items[cluster_id].append((item_id, popularity))

        # Sort items within each cluster
        for cluster_id in tqdm(cluster_items, desc="Sorting items within clusters"):
            cluster_items[cluster_id].sort(key=lambda x: x[1], reverse=True)
        
        # Select diverse recommendations
        recommendations = []
        items_per_cluster = max(1, n_recommendations // self.n_clusters)
        
        for cluster_id in tqdm(range(self.n_clusters), desc="Selecting diverse recommendations"):
            cluster_recommendations = cluster_items[cluster_id][:items_per_cluster]
            recommendations.extend(cluster_recommendations)
        
        # Sort by popularity and return top N
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:n_recommendations]

    def _content_based_item_recommendations(self, item_id: int, n_recommendations: int) -> List[Tuple[int, float]]:
        """
        Content-based recommendations for new items
        """
        if item_id not in self.movie_id_to_idx:
            print(f"⚠️  Item {item_id} not found in TF-IDF features")
            return []

        # Get item's TF-IDF features
        item_idx = self.movie_id_to_idx[item_id]
        item_features = self.tfidf_features[item_idx].reshape(1, -1)

        # Calculate similarity with all items
        similarities = cosine_similarity(item_features, self.tfidf_features)[0]

        # Get most similar items (excluding self)
        similar_indices = np.argsort(similarities)[::-1][1:n_recommendations+1]

        recommendations = []
        for idx in similar_indices:
            if similarities[idx] >= self.min_similarity:
                similar_item_id = self.idx_to_movie_id[idx]
                recommendations.append((similar_item_id, similarities[idx]))
        
        return recommendations

    def hybrid_cold_start_recommendations(self, user_id: int, n_recommendations: int = 10,
                                        weights: Dict[str, float] = None) -> List[Tuple[int, float]]:
        """
        Hybrid approach combining multiple cold start strategies
        """

        if weights is None:
            weights = {'popularity': 0.4, 'cluster': 0.3, 'diversity': 0.3}

        print(f"🔄 Generating hybrid recommendations for user {user_id}...")
        
        # Get recommendations from different methods
        popularity_recs = dict(self._popularity_recommendations(n_recommendations * 2))
        cluster_recs = dict(self._cluster_based_recommendations(user_id, n_recommendations * 2))
        diversity_recs = dict(self._diversity_recommendations(n_recommendations * 2))

        # Combine scores
        combined_scores = {}

        for item_id in set(list(popularity_recs.keys()) + 
                          list(cluster_recs.keys()) + 
                          list(diversity_recs.keys())):
            score = 0
            if item_id in popularity_recs:
                score += weights['popularity'] * popularity_recs[item_id]
            if item_id in cluster_recs:
                score += weights['cluster'] * cluster_recs[item_id]
            if item_id in diversity_recs:
                score += weights['diversity'] * diversity_recs[item_id]
            
            combined_scores[item_id] = score
        
        # Sort by combined score
        sorted_recommendations = sorted(combined_scores.items(), 
                                      key=lambda x: x[1], reverse=True)
        
        return sorted_recommendations[:n_recommendations]

    def progressive_cold_start(self, user_id: int, initial_ratings: Optional[List[Tuple[int, float]]] = None,
                             n_recommendations: int = 10) -> List[Tuple[int, float]]:
        """
        Progressive cold start that adapts as user provides initial ratings
        """
        print(f"📈 Progressive cold start for user {user_id}...")

        if initial_ratings is None or len(initial_ratings) == 0:
            # No ratings yet, use diversity recommendations
            print("   No initial ratings - using diversity recommendations")
            return self._diversity_recommendations(n_recommendations)
        elif len(initial_ratings) < 3:
            # Few ratings - use hybrid approach
            print(f"   {len(initial_ratings)} initial ratings - using hybrid approach")
            return self.hybrid_cold_start_recommendations(user_id, n_recommendations)
        
        else:
            # Enough ratings - use content-based approach
            print(f"   {len(initial_ratings)} initial ratings - using content-based approach")
            return self._content_based_from_ratings(user_id, initial_ratings, n_recommendations)

    def _content_based_from_ratings(self, user_id: int, ratings: List[Tuple[int, float]], 
                                   n_recommendations: int) -> List[Tuple[int, float]]:
        """
        Content-based recommendations based on user's initial ratings
        """
        # Calculate user's content profile from rated items
        user_profile = np.zeros(self.tfidf_features.shape[1])
        
        for item_id, rating in ratings:
            if item_id in self.movie_id_to_idx:
                item_idx = self.movie_id_to_idx[item_id]
                user_profile += rating * self.tfidf_features[item_idx]
        
        # Normalize user profile
        if np.sum(user_profile) > 0:
            user_profile = user_profile / np.sum(user_profile)
        
        # Calculate similarity with all items
        user_profile_reshaped = user_profile.reshape(1, -1)
        similarities = cosine_similarity(user_profile_reshaped, self.tfidf_features)[0]
        
        # Get most similar items (excluding already rated)
        rated_items = set(item_id for item_id, _ in ratings)
        
        recommendations = []
        for idx in np.argsort(similarities)[::-1]:
            item_id = self.idx_to_movie_id[idx]
            if item_id not in rated_items and similarities[idx] >= self.min_similarity:
                recommendations.append((item_id, similarities[idx]))
                if len(recommendations) >= n_recommendations:
                    break
        
        return recommendations
    
    def evaluate_cold_start(self, test_users: List[int], test_items: List[int], 
                           movies: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate cold start solutions
        """
        print("📊 Evaluating cold start solutions...")
        
        # Test user cold start
        user_results = self._evaluate_user_cold_start(test_users, movies)
        
        # Test item cold start
        item_results = self._evaluate_item_cold_start(test_items)
        
        # Combine results
        results = {
            'user_cold_start': user_results,
            'item_cold_start': item_results,
            'overall_score': (user_results['diversity_score'] + item_results['similarity_score']) / 2
        }
        
        return results
    
    def _evaluate_user_cold_start(self, test_users: List[int], movies: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate user cold start by measuring recommendation diversity
        """
        print("   Evaluating user cold start...")
        
        all_recommendations = []
        
        for user_id in test_users[:10]:  # Test first 10 users
            recs = self.hybrid_cold_start_recommendations(user_id, 10)
            all_recommendations.extend([item_id for item_id, _ in recs])
        
        # Calculate diversity (unique items / total recommendations)
        unique_items = len(set(all_recommendations))
        total_recommendations = len(all_recommendations)
        diversity_score = unique_items / total_recommendations if total_recommendations > 0 else 0
        
        # Calculate genre diversity
        genre_diversity = self._calculate_genre_diversity(all_recommendations, movies)
        
        return {
            'diversity_score': diversity_score,
            'genre_diversity': genre_diversity,
            'total_recommendations': total_recommendations,
            'unique_items': unique_items
        }
    
    def _evaluate_item_cold_start(self, test_items: List[int]) -> Dict[str, float]:
        """
        Evaluate item cold start by measuring similarity quality
        """
        print("   Evaluating item cold start...")
        
        similarity_scores = []
        
        for item_id in test_items[:10]:  # Test first 10 items
            recs = self._content_based_item_recommendations(item_id, 5)
            if recs:
                avg_similarity = np.mean([score for _, score in recs])
                similarity_scores.append(avg_similarity)
        
        avg_similarity_score = np.mean(similarity_scores) if similarity_scores else 0
        
        return {
            'similarity_score': avg_similarity_score,
            'n_items_tested': len(similarity_scores),
            'avg_recommendations_per_item': len(similarity_scores) / 10 if similarity_scores else 0
        }
    
    def _calculate_genre_diversity(self, item_ids: List[int], movies: pd.DataFrame) -> float:
        """
        Calculate genre diversity of recommended items
        """
        genres = []
        
        for item_id in item_ids:
            movie = movies[movies['movieId'] == item_id]
            if not movie.empty:
                movie_genres = str(movie.iloc[0].get('genres', '')).split('|')
                genres.extend(movie_genres)
        
        unique_genres = len(set(genres))
        total_genres = len(genres)
        
        return unique_genres / total_genres if total_genres > 0 else 0
    
    def compare_cold_start_methods(self, user_id: int, n_recommendations: int = 10) -> Dict[str, List[Tuple[int, float]]]:
        """
        Compare different cold start methods for a user
        """
        print(f"🔍 Comparing cold start methods for user {user_id}...")
        
        methods = {
            'popularity': self._popularity_recommendations(n_recommendations),
            'cluster': self._cluster_based_recommendations(user_id, n_recommendations),
            'diversity': self._diversity_recommendations(n_recommendations),
            'hybrid': self.hybrid_cold_start_recommendations(user_id, n_recommendations),
            'progressive': self.progressive_cold_start(user_id, None, n_recommendations)
        }
        
        return methods


def run_cold_start_demo():
    """
    Run a demonstration of cold start solutions
    """
    print("🚀 COLD START SOLUTIONS DEMO")
    print("="*50)
    
    # Load data
    print("📂 Loading MovieLens data...")
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
    
    # Extract year from title
    movies['year'] = movies['title'].str.extract(r'\((\d{4})\)')
    
    print(f"Ratings: {len(ratings)} records")
    print(f"Movies: {len(movies)} records")
    
    # Initialize cold start system
    print("\n🔧 Initializing cold start solutions...")
    cold_start = ColdStartSolutions(n_clusters=8, min_similarity=0.1)
    
    # Train the system
    print("\n🎯 Training cold start system...")
    cold_start.fit(ratings, movies)
    
    # Test different cold start methods
    print("\n Testing cold start methods...")
    test_user = 999999  # New user ID
    
    methods_comparison = cold_start.compare_cold_start_methods(test_user, 5)
    
    for method_name, recommendations in methods_comparison.items():
        print(f"\n {method_name.upper()} recommendations:")
        for item_id, score in recommendations:
            movie_title = movies[movies['movieId'] == item_id]['title'].iloc[0] if not movies[movies['movieId'] == item_id].empty else f"Movie {item_id}"
            print(f"   {movie_title}: {score:.3f}")
    
    # Test progressive cold start
    print("\n📈 Testing progressive cold start...")
    initial_ratings = [(1, 5.0), (2, 4.0)]  # User rates two movies
    progressive_recs = cold_start.progressive_cold_start(test_user, initial_ratings, 5)
    
    print("Progressive recommendations after 2 ratings:")
    for item_id, score in progressive_recs:
        movie_title = movies[movies['movieId'] == item_id]['title'].iloc[0] if not movies[movies['movieId'] == item_id].empty else f"Movie {item_id}"
        print(f"   {movie_title}: {score:.3f}")
    
    # Test item cold start
    print("\n🎭 Testing item cold start...")
    test_item = 999999  # New item ID
    item_recs = cold_start.recommend_for_new_item(1, 5)  # Use existing item for demo
    
    print("Similar items to Movie 1:")
    for item_id, similarity in item_recs:
        movie_title = movies[movies['movieId'] == item_id]['title'].iloc[0] if not movies[movies['movieId'] == item_id].empty else f"Movie {item_id}"
        print(f"   {movie_title}: {similarity:.3f}")
    
    # Evaluate cold start solutions
    print("\n📊 Evaluating cold start solutions...")
    test_users = [999999, 999998, 999997]  # Simulate new users
    test_items = [999999, 999998, 999997]  # Simulate new items
    
    evaluation_results = cold_start.evaluate_cold_start(test_users, test_items, movies)
    
    print(f"\n Evaluation Results:")
    print(f"   User Cold Start Diversity: {evaluation_results['user_cold_start']['diversity_score']:.3f}")
    print(f"   Item Cold Start Similarity: {evaluation_results['item_cold_start']['similarity_score']:.3f}")
    print(f"   Overall Score: {evaluation_results['overall_score']:.3f}")
    
    print("\n✅ Cold start solutions demo completed!")


if __name__ == "__main__":
    run_cold_start_demo()
    
        

        
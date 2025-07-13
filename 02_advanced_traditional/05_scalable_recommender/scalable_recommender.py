import numpy as np
import pandas as pd
import time
import gc
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import pickle
import os
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class ScalableRecommender:
    """
    Scalable recommendation system with chunking, incremental learning,
    and memory optimization techniques.
    """
    
    def __init__(self, chunk_size: int = 10000, memory_limit_gb: float = 2.0):
        self.chunk_size = chunk_size
        self.memory_limit_gb = memory_limit_gb
        self.user_factors = {}
        self.item_factors = {}
        self.user_item_matrix = None
        self.similarity_cache = {}
        self.training_history = []
        
    def _get_memory_usage(self) -> float:
        """Get current memory usage in GB"""
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024 / 1024

    def _process_in_chunks(self, data: pd.DataFrame, process_func) -> List:
        """
        Process data in chunks to manage memory usage
        
        Args:
            data: DataFrame to process
            process_func: Function to apply to each chunk
            
        Returns:
            List of results from processing each chunk
        """
        results = []
        total_chunks = len(data) // self.chunk_size + 1
        
        print(f"Processing {len(data)} records in {total_chunks} chunks...")
        
        for i in tqdm(range(0, len(data), self.chunk_size), desc="Processing chunks"):
            chunk = data.iloc[i:i + self.chunk_size]
            
            # Check memory usage
            if self._get_memory_usage() > self.memory_limit_gb:
                print(f"  Memory usage high ({self._get_memory_usage():.2f}GB), forcing garbage collection...")
                gc.collect()
            
            # Process chunk
            chunk_result = process_func(chunk)
            results.append(chunk_result)
            
            # Clear chunk from memory
            del chunk
            
        return results
    
    def _build_user_item_matrix_chunked(self, ratings: pd.DataFrame) -> pd.DataFrame:
        """
        Build user-item matrix using chunked processing
        """
        print("🏗️  Building user-item matrix with chunked processing...")
        
        # Get unique users and items
        unique_users = ratings['userId'].unique()
        unique_items = ratings['movieId'].unique()
        
        # Create user and item mappings
        user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
        item_to_idx = {item: idx for idx, item in enumerate(unique_items)}
        
        # Initialize empty matrix
        matrix = pd.DataFrame(0, 
                            index=range(len(unique_users)), 
                            columns=range(len(unique_items)))
        
        def process_chunk(chunk):
            """Process a chunk of ratings"""
            chunk_matrix = pd.DataFrame(0, 
                                      index=range(len(unique_users)), 
                                      columns=range(len(unique_items)))
            
            for _, row in chunk.iterrows():
                user_idx = user_to_idx[row['userId']]
                item_idx = item_to_idx[row['movieId']]
                chunk_matrix.iloc[user_idx, item_idx] = row['rating']
            
            return chunk_matrix
        
        # Process in chunks
        chunk_matrices = self._process_in_chunks(ratings, process_chunk)
        
        # Combine results
        final_matrix = pd.concat(chunk_matrices, axis=0).groupby(level=0).sum()
        
        # Store mappings
        self.user_to_idx = user_to_idx
        self.item_to_idx = item_to_idx
        self.idx_to_user = {idx: user for user, idx in user_to_idx.items()}
        self.idx_to_item = {idx: item for item, idx in item_to_idx.items()}
        
        print(f"✅ Matrix built: {final_matrix.shape[0]} users × {final_matrix.shape[1]} items")
        return final_matrix
    
    def _compute_similarity_with_cache(self, matrix: pd.DataFrame, similarity_type: str = 'cosine') -> np.ndarray:
        """
        Compute similarity matrix with caching to avoid recomputation
        """
        cache_key = f"{similarity_type}_{matrix.shape[0]}_{matrix.shape[1]}"
        
        # Check if already computed
        if cache_key in self.similarity_cache:
            print(f"📋 Using cached {similarity_type} similarity matrix...")
            return self.similarity_cache[cache_key]
        
        print(f" Computing {similarity_type} similarity matrix...")
        
        # Compute similarity in chunks for large matrices
        if matrix.shape[0] > 1000:  # Large matrix, use chunked computation
            similarity_matrix = self._compute_similarity_chunked(matrix, similarity_type)
        else:
            # Small matrix, compute directly
            if similarity_type == 'cosine':
                similarity_matrix = cosine_similarity(matrix)
            else:
                similarity_matrix = np.corrcoef(matrix)
        
        # Cache the result
        self.similarity_cache[cache_key] = similarity_matrix
        print(f"✅ Similarity matrix computed and cached")
        
        return similarity_matrix
    
    def _compute_similarity_chunked(self, matrix: pd.DataFrame, similarity_type: str) -> np.ndarray:
        """
        Compute similarity matrix in chunks for large datasets
        """
        n_rows = matrix.shape[0]
        chunk_size = min(500, n_rows // 4)  # Adaptive chunk size
        similarity_matrix = np.zeros((n_rows, n_rows))
        
        for i in tqdm(range(0, n_rows, chunk_size), desc="Computing similarity chunks"):
            end_i = min(i + chunk_size, n_rows)
            chunk_i = matrix.iloc[i:end_i]
            
            for j in range(0, n_rows, chunk_size):
                end_j = min(j + chunk_size, n_rows)
                chunk_j = matrix.iloc[j:end_j]
                
                if similarity_type == 'cosine':
                    chunk_similarity = cosine_similarity(chunk_i, chunk_j)
                else:
                    chunk_similarity = np.corrcoef(chunk_i, chunk_j)
                
                similarity_matrix[i:end_i, j:end_j] = chunk_similarity
        
        return similarity_matrix
    
    def incremental_update(self, new_ratings: pd.DataFrame):
        """
        Incrementally update the model with new ratings
        """
        print("🔄 Performing incremental update...")
        
        if self.user_item_matrix is None:
            print("⚠️  No existing model found. Building from scratch...")
            self.fit(new_ratings)
            return
        
        # Get new users and items
        existing_users = set(self.user_to_idx.keys())
        existing_items = set(self.item_to_idx.keys())
        
        new_users = set(new_ratings['userId'].unique()) - existing_users
        new_items = set(new_ratings['movieId'].unique()) - existing_items
        
        print(f"📊 New users: {len(new_users)}, New items: {len(new_items)}")
        
        # Update mappings
        for user in new_users:
            self.user_to_idx[user] = len(self.user_to_idx)
            self.idx_to_user[len(self.idx_to_user)] = user
        
        for item in new_items:
            self.item_to_idx[item] = len(self.item_to_idx)
            self.idx_to_item[len(self.idx_to_item)] = item
        
        # Expand matrix
        new_matrix = pd.DataFrame(0, 
                                index=range(len(self.user_to_idx)), 
                                columns=range(len(self.item_to_idx)))
        
        # Copy existing data
        new_matrix.iloc[:self.user_item_matrix.shape[0], :self.user_item_matrix.shape[1]] = self.user_item_matrix
        
        # Add new ratings
        for _, row in new_ratings.iterrows():
            user_idx = self.user_to_idx[row['userId']]
            item_idx = self.item_to_idx[row['movieId']]
            new_matrix.iloc[user_idx, item_idx] = row['rating']
        
        self.user_item_matrix = new_matrix
        
        # Clear similarity cache (will be recomputed when needed)
        self.similarity_cache.clear()
        
        print("✅ Incremental update completed")
    
    def fit(self, ratings: pd.DataFrame):
        """
        Fit the scalable recommender system
        """
        print("🚀 Training scalable recommender system...")
        start_time = time.time()
        
        # Build user-item matrix with chunked processing
        self.user_item_matrix = self._build_user_item_matrix_chunked(ratings)
        
        # Compute user similarities with caching
        self.user_similarity = self._compute_similarity_with_cache(self.user_item_matrix, 'cosine')
        
        # Compute item similarities with caching
        self.item_similarity = self._compute_similarity_with_cache(self.user_item_matrix.T, 'cosine')
        
        training_time = time.time() - start_time
        self.training_history.append({
            'timestamp': time.time(),
            'training_time': training_time,
            'matrix_shape': self.user_item_matrix.shape,
            'memory_usage': self._get_memory_usage()
        })
        
        print(f"✅ Training completed in {training_time:.2f} seconds")
        print(f"📊 Memory usage: {self._get_memory_usage():.2f} GB")
    
    def recommend_for_user(self, user_id: int, n_recommendations: int = 10, 
                          method: str = 'collaborative') -> List[Tuple[int, float]]:
        """
        Generate recommendations for a user using scalable methods
        """
        if user_id not in self.user_to_idx:
            print(f"⚠️  User {user_id} not found in training data")
            return []
        
        user_idx = self.user_to_idx[user_id]
        
        if method == 'collaborative':
            return self._collaborative_recommendations(user_idx, n_recommendations)
        elif method == 'content':
            return self._content_recommendations(user_idx, n_recommendations)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _collaborative_recommendations(self, user_idx: int, n_recommendations: int) -> List[Tuple[int, float]]:
        """
        Generate collaborative filtering recommendations
        """
        # Get user's ratings
        user_ratings = self.user_item_matrix.iloc[user_idx]
        rated_items = user_ratings[user_ratings > 0].index.tolist()
        
        if len(rated_items) == 0:
            print("⚠️  User has no ratings")
            return []
        
        # Find similar users (top 20)
        user_similarities = self.user_similarity[user_idx]
        similar_users = np.argsort(user_similarities)[::-1][1:21]  # Exclude self
        
        # Calculate predicted ratings
        predictions = {}
        
        for item_idx in range(self.user_item_matrix.shape[1]):
            if item_idx in rated_items:
                continue  # Skip already rated items
            
            # Get ratings for this item from similar users
            item_ratings = []
            similarities = []
            
            for similar_user_idx in similar_users:
                rating = self.user_item_matrix.iloc[similar_user_idx, item_idx]
                if rating > 0:
                    item_ratings.append(rating)
                    similarities.append(user_similarities[similar_user_idx])
            
            if len(item_ratings) > 0:
                # Weighted average prediction
                weighted_sum = sum(r * s for r, s in zip(item_ratings, similarities))
                similarity_sum = sum(similarities)
                prediction = weighted_sum / similarity_sum
                predictions[item_idx] = prediction
        
        # Sort by prediction score
        sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        
        # Convert back to original item IDs
        recommendations = []
        for item_idx, score in sorted_predictions[:n_recommendations]:
            original_item_id = self.idx_to_item[item_idx]
            recommendations.append((original_item_id, score))
        
        return recommendations
    
    def _content_recommendations(self, user_idx: int, n_recommendations: int) -> List[Tuple[int, float]]:
        """
        Generate content-based recommendations using item similarities
        """
        # Get user's rated items
        user_ratings = self.user_item_matrix.iloc[user_idx]
        rated_items = user_ratings[user_ratings > 0]
        
        if len(rated_items) == 0:
            print("⚠️  User has no ratings")
            return []
        
        # Calculate item scores based on similarity to rated items
        item_scores = {}
        
        for item_idx in range(self.user_item_matrix.shape[1]):
            if item_idx in rated_items.index:
                continue  # Skip already rated items
            
            score = 0
            for rated_item_idx, rating in rated_items.items():
                similarity = self.item_similarity[item_idx, rated_item_idx]
                score += similarity * rating
            
            if score > 0:
                item_scores[item_idx] = score
        
        # Sort by score
        sorted_scores = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Convert back to original item IDs
        recommendations = []
        for item_idx, score in sorted_scores[:n_recommendations]:
            original_item_id = self.idx_to_item[item_idx]
            recommendations.append((original_item_id, score))
        
        return recommendations
    
    def evaluate(self, test_ratings: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate the scalable recommender system
        """
        print("📊 Evaluating scalable recommender...")
        
        predictions = []
        actuals = []
        
        for _, row in tqdm(test_ratings.iterrows(), desc="Evaluating predictions"):
            user_id = row['userId']
            item_id = row['movieId']
            actual_rating = row['rating']
            
            if user_id in self.user_to_idx and item_id in self.item_to_idx:
                user_idx = self.user_to_idx[user_id]
                item_idx = self.item_to_idx[item_id]
                
                # Get user's similar users
                user_similarities = self.user_similarity[user_idx]
                similar_users = np.argsort(user_similarities)[::-1][1:11]  # Top 10
                
                # Calculate prediction
                item_ratings = []
                similarities = []
                
                for similar_user_idx in similar_users:
                    rating = self.user_item_matrix.iloc[similar_user_idx, item_idx]
                    if rating > 0:
                        item_ratings.append(rating)
                        similarities.append(user_similarities[similar_user_idx])
                
                if len(item_ratings) > 0:
                    weighted_sum = sum(r * s for r, s in zip(item_ratings, similarities))
                    similarity_sum = sum(similarities)
                    predicted_rating = weighted_sum / similarity_sum
                    
                    predictions.append(predicted_rating)
                    actuals.append(actual_rating)
        
        if len(predictions) == 0:
            print("⚠️  No predictions could be made")
            return {}
        
        # Calculate metrics
        rmse = np.sqrt(np.mean((np.array(predictions) - np.array(actuals)) ** 2))
        mae = np.mean(np.abs(np.array(predictions) - np.array(actuals)))
        
        results = {
            'rmse': rmse,
            'mae': mae,
            'n_predictions': len(predictions)
        }
        
        print(f"✅ Evaluation completed:")
        print(f"   RMSE: {rmse:.4f}")
        print(f"   MAE: {mae:.4f}")
        print(f"   Predictions: {len(predictions)}")
        
        return results
    
    def save_model(self, filepath: str):
        """
        Save the model to disk
        """
        print(f"💾 Saving model to {filepath}...")
        
        model_data = {
            'user_item_matrix': self.user_item_matrix,
            'user_similarity': self.user_similarity,
            'item_similarity': self.item_similarity,
            'user_to_idx': self.user_to_idx,
            'item_to_idx': self.item_to_idx,
            'idx_to_user': self.idx_to_user,
            'idx_to_item': self.idx_to_item,
            'similarity_cache': self.similarity_cache,
            'training_history': self.training_history,
            'chunk_size': self.chunk_size,
            'memory_limit_gb': self.memory_limit_gb
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print("✅ Model saved successfully")
    
    def load_model(self, filepath: str):
        """
        Load the model from disk
        """
        print(f" Loading model from {filepath}...")
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.user_item_matrix = model_data['user_item_matrix']
        self.user_similarity = model_data['user_similarity']
        self.item_similarity = model_data['item_similarity']
        self.user_to_idx = model_data['user_to_idx']
        self.item_to_idx = model_data['item_to_idx']
        self.idx_to_user = model_data['idx_to_user']
        self.idx_to_item = model_data['idx_to_item']
        self.similarity_cache = model_data['similarity_cache']
        self.training_history = model_data['training_history']
        self.chunk_size = model_data['chunk_size']
        self.memory_limit_gb = model_data['memory_limit_gb']
        
        print("✅ Model loaded successfully")
    
    def get_performance_stats(self) -> Dict:
        """
        Get performance statistics
        """
        if not self.training_history:
            return {}
        
        latest = self.training_history[-1]
        return {
            'matrix_shape': latest['matrix_shape'],
            'training_time': latest['training_time'],
            'memory_usage': latest['memory_usage'],
            'cache_size': len(self.similarity_cache)
        }


def run_scalable_recommender_demo():
    """
    Run a demonstration of the scalable recommender system
    """
    print("🚀 SCALABLE RECOMMENDER SYSTEM DEMO")
    print("="*50)
    
    # Load data
    print("📂 Loading MovieLens data...")
    ratings = pd.read_csv('../../01_fundamentals/01_movielens_exploration/ml-1m/ratings.dat', 
                         sep='::', 
                         names=['userId', 'movieId', 'rating', 'timestamp'],
                         engine='python',
                         encoding='latin-1')
    
    # Use a larger subset to demonstrate scalability
    print(f"Original dataset: {len(ratings)} ratings")
    ratings_subset = ratings.sample(n=min(100000, len(ratings)), random_state=42)
    print(f"Using subset: {len(ratings_subset)} ratings")
    
    # Split data
    train_size = 0.8
    train_ratings = ratings_subset.sample(frac=train_size, random_state=42)
    test_ratings = ratings_subset.drop(train_ratings.index)
    
    print(f"Training set: {len(train_ratings)} ratings")
    print(f"Test set: {len(test_ratings)} ratings")
    
    # Initialize scalable recommender
    print("\n🔧 Initializing scalable recommender...")
    recommender = ScalableRecommender(chunk_size=5000, memory_limit_gb=1.5)
    
    # Train the model
    print("\n🎯 Training model...")
    recommender.fit(train_ratings)
    
    # Get performance stats
    stats = recommender.get_performance_stats()
    print(f"\n📊 Performance Statistics:")
    print(f"   Matrix shape: {stats['matrix_shape']}")
    print(f"   Training time: {stats['training_time']:.2f}s")
    print(f"   Memory usage: {stats['memory_usage']:.2f}GB")
    print(f"   Cache size: {stats['cache_size']}")
    
    # Test recommendations
    print("\n🎬 Testing recommendations...")
    test_user = train_ratings['userId'].iloc[0]
    
    collaborative_recs = recommender.recommend_for_user(test_user, 5, 'collaborative')
    content_recs = recommender.recommend_for_user(test_user, 5, 'content')
    
    print(f"\n📋 Collaborative recommendations for user {test_user}:")
    for item_id, score in collaborative_recs:
        print(f"   Movie {item_id}: {score:.3f}")
    
    print(f"\n📋 Content-based recommendations for user {test_user}:")
    for item_id, score in content_recs:
        print(f"   Movie {item_id}: {score:.3f}")
    
    # Evaluate model
    print("\n📊 Evaluating model...")
    evaluation_results = recommender.evaluate(test_ratings)
    
    # Test incremental update
    print("\n🔄 Testing incremental update...")
    new_ratings = test_ratings.sample(n=min(1000, len(test_ratings)), random_state=123)
    recommender.incremental_update(new_ratings)
    
    # Save and load model
    print("\n💾 Testing model persistence...")
    recommender.save_model('scalable_recommender_model.pkl')
    
    new_recommender = ScalableRecommender()
    new_recommender.load_model('scalable_recommender_model.pkl')
    
    print("✅ Scalable recommender demo completed!")


if __name__ == "__main__":
    run_scalable_recommender_demo()
    

 
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import time


class OptimizationMethods:
    """
    Compare different optimization methods for matrix factorization
    """

    def __init__(self, n_factors=50, learning_rate=0.01, n_epochs=100):
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.history = {}

    def create_user_item_matrix(self, ratings):
        """Create user-item matrix from ratings"""
        return ratings.pivot_table(
            index="userId", columns="movieId", values="rating", fill_value=0
        )

    def sgd_optimization(self, user_item_matrix, verbose=True):
        """
        Basic Stochastic Gradient Descent for matrix factorization
        """
        print("Training with SGD...")

        # Initialize matrices
        n_users, n_items = user_item_matrix.shape
        U = np.random.normal(0, 0.1, (n_users, self.n_factors))
        V = np.random.normal(0, 0.1, (n_items, self.n_factors))

        # Get non-zero ratings
        ratings = []
        for i in tqdm(range(n_users), desc="Processing users"):
            for j in range(n_items):
                if user_item_matrix.iloc[i, j] > 0:
                    ratings.append((i, j, user_item_matrix.iloc[i, j]))

        losses = []
        start_time = time.time()

        for epoch in tqdm(range(self.n_epochs), desc="Training epochs"):
            epoch_loss = 0
            np.random.shuffle(ratings)  # shuffle for stochasticity

            for user_idx, item_idx, rating in tqdm(
                ratings, desc="Processing ratings", leave=False
            ):
                # Forward pass
                prediction = np.dot(U[user_idx], V[item_idx])
                error = rating - prediction

                # Backward pass
                U[user_idx] += self.learning_rate * error * V[item_idx]
                V[item_idx] += self.learning_rate * error * U[user_idx]

                epoch_loss += error**2

            avg_loss = epoch_loss / len(ratings)
            losses.append(avg_loss)

            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")

        training_time = time.time() - start_time
        print(f"SGD training completed in {training_time:.2f} seconds")

        self.history["sgd"] = {"losses": losses, "time": training_time, "U": U, "V": V}

        return U, V

    def sgd_momentum_optimization(self, user_item_matrix, momentum=0.9, verbose=True):
        """
        SGD with Momentum for better convergence
        """
        print("Training with SGD + Momentum...")

        # Initialize matrices
        n_users, n_items = user_item_matrix.shape
        U = np.random.normal(0, 0.1, (n_users, self.n_factors))
        V = np.random.normal(0, 0.1, (n_items, self.n_factors))

        # Initialize momentum
        U_momentum = np.zeros_like(U)
        V_momentum = np.zeros_like(V)

        # Get non-zero ratings
        ratings = []
        for i in range(n_users):
            for j in range(n_items):
                if user_item_matrix.iloc[i, j] > 0:
                    ratings.append((i, j, user_item_matrix.iloc[i, j]))

        losses = []
        start_time = time.time()

        for epoch in tqdm(range(self.n_epochs), desc="Training epochs"):
            epoch_loss = 0
            np.random.shuffle(ratings)

            for user_idx, item_idx, rating in ratings:
                # Forward pass
                prediction = np.dot(U[user_idx], V[item_idx])
                error = rating - prediction

                # Calculate gradients
                U_grad = error * V[item_idx]
                V_grad = error * U[user_idx]

                # Update momentum
                U_momentum[user_idx] = (
                    momentum * U_momentum[user_idx] + self.learning_rate * U_grad
                )
                V_momentum[item_idx] = (
                    momentum * V_momentum[item_idx] + self.learning_rate * V_grad
                )

                # Update parameters
                U[user_idx] += U_momentum[user_idx]
                V[item_idx] += V_momentum[item_idx]

                epoch_loss += error**2

            avg_loss = epoch_loss / len(ratings)
            losses.append(avg_loss)

            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")

        training_time = time.time() - start_time
        print(f"SGD + Momentum training completed in {training_time:.2f} seconds")

        self.history["sgd_momentum"] = {
            "losses": losses,
            "time": training_time,
            "U": U,
            "V": V,
        }

        return U, V

    def adaptive_sgd_optimization(
        self, user_item_matrix, beta1=0.9, beta2=0.999, epsilon=1e-8, verbose=True
    ):
        """
        SGD with adaptive learning rates (Adam-like)
        """
        print("Training with Adaptive SGD...")

        # Initialize matrices
        n_users, n_items = user_item_matrix.shape
        U = np.random.normal(0, 0.1, (n_users, self.n_factors))
        V = np.random.normal(0, 0.1, (n_items, self.n_factors))

        # Initialize adaptive parameters
        U_m = np.zeros_like(U)  # First moment
        U_v = np.zeros_like(U)  # Second moment
        V_m = np.zeros_like(V)  # First moment
        V_v = np.zeros_like(V)  # Second moment

        # Get non-zero ratings
        ratings = []
        for i in range(n_users):
            for j in range(n_items):
                if user_item_matrix.iloc[i, j] > 0:
                    ratings.append((i, j, user_item_matrix.iloc[i, j]))

        losses = []
        start_time = time.time()

        for epoch in tqdm(range(self.n_epochs), desc="Training epochs"):
            epoch_loss = 0
            np.random.shuffle(ratings)

            for user_idx, item_idx, rating in ratings:
                # Forward pass
                prediction = np.dot(U[user_idx], V[item_idx])
                error = rating - prediction

                # Calculate gradients
                U_grad = error * V[item_idx]
                V_grad = error * U[user_idx]

                # Update first moment (biased)
                U_m[user_idx] = beta1 * U_m[user_idx] + (1 - beta1) * U_grad
                V_m[item_idx] = beta1 * V_m[item_idx] + (1 - beta1) * V_grad

                # Update second moment (biased)
                U_v[user_idx] = beta2 * U_v[user_idx] + (1 - beta2) * (U_grad**2)
                V_v[item_idx] = beta2 * V_v[item_idx] + (1 - beta2) * (V_grad**2)

                # Bias correction
                t = epoch * len(ratings) + 1
                U_m_corrected = U_m[user_idx] / (1 - beta1**t)
                V_m_corrected = V_m[item_idx] / (1 - beta1**t)
                U_v_corrected = U_v[user_idx] / (1 - beta2**t)
                V_v_corrected = V_v[item_idx] / (1 - beta2**t)

                # Update parameters with adaptive learning rate
                U[user_idx] += (
                    self.learning_rate
                    * U_m_corrected
                    / (np.sqrt(U_v_corrected) + epsilon)
                )
                V[item_idx] += (
                    self.learning_rate
                    * V_m_corrected
                    / (np.sqrt(V_v_corrected) + epsilon)
                )

                epoch_loss += error**2

            avg_loss = epoch_loss / len(ratings)
            losses.append(avg_loss)

            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")

        training_time = time.time() - start_time
        print(f"Adaptive SGD training completed in {training_time:.2f} seconds")

        self.history["adaptive_sgd"] = {
            "losses": losses,
            "time": training_time,
            "U": U,
            "V": V,
        }

        return U, V

    def als_optimization(self, user_item_matrix, lambda_reg=0.1, verbose=True):
        """
        Alternating Least Squares for matrix factorization
        """
        print("Training with ALS...")

        # Initialize matrices
        n_users, n_items = user_item_matrix.shape
        U = np.random.normal(0, 0.1, (n_users, self.n_factors))
        V = np.random.normal(0, 0.1, (n_items, self.n_factors))

        # Create sparse matrix for efficiency
        ratings_matrix = user_item_matrix.values
        mask = ratings_matrix > 0

        losses = []
        start_time = time.time()

        for epoch in tqdm(range(self.n_epochs), desc="Training epochs"):
            epoch_loss = 0

            # Fix V, solve for U
            for i in range(n_users):
                # Get items rated by user i
                rated_items = np.where(mask[i])[0]
                if len(rated_items) == 0:
                    continue

                # Create submatrices
                V_sub = V[rated_items]
                R_sub = ratings_matrix[i, rated_items]

                # Solve: (V_sub.T @ V_sub + λI) @ U[i] = V_sub.T @ R_sub
                A = V_sub.T @ V_sub + lambda_reg * np.eye(self.n_factors)
                b = V_sub.T @ R_sub
                U[i] = np.linalg.solve(A, b)

            # Fix U, solve for V
            for j in range(n_items):
                # Get users who rated item j
                rated_users = np.where(mask[:, j])[0]
                if len(rated_users) == 0:
                    continue

                # Create submatrices
                U_sub = U[rated_users]
                R_sub = ratings_matrix[rated_users, j]

                # Solve: (U_sub.T @ U_sub + λI) @ V[j] = U_sub.T @ R_sub
                A = U_sub.T @ U_sub + lambda_reg * np.eye(self.n_factors)
                b = U_sub.T @ R_sub
                V[j] = np.linalg.solve(A, b)

            # Calculate loss
            predictions = U @ V.T
            error = ratings_matrix - predictions
            epoch_loss = np.mean(error[mask] ** 2)
            losses.append(epoch_loss)

            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {epoch_loss:.4f}")

        training_time = time.time() - start_time
        print(f"ALS training completed in {training_time:.2f} seconds")

        self.history["als"] = {"losses": losses, "time": training_time, "U": U, "V": V}

        return U, V

    def evaluate_method(self, U, V, test_ratings, method_name):
        """
        Evaluate a trained model on test data
        """
        predictions = []
        actuals = []

        for _, row in tqdm(
            test_ratings.iterrows(), desc="Evaluating predictions", leave=False
        ):
            user_idx = row["user_idx"]
            item_idx = row["item_idx"]
            actual_rating = row["rating"]

            if user_idx < U.shape[0] and item_idx < V.shape[0]:
                predicted_rating = np.dot(U[user_idx], V[item_idx])
                predictions.append(predicted_rating)
                actuals.append(actual_rating)

        if len(predictions) > 0:
            rmse = np.sqrt(mean_squared_error(actuals, predictions))
            mae = np.mean(np.abs(np.array(predictions) - np.array(actuals)))
            return {"rmse": rmse, "mae": mae}
        else:
            return {"rmse": float("inf"), "mae": float("inf")}

    def compare_optimization_methods(self, ratings, test_size=0.2):
        """
        Compare all optimization methods
        """
        print("=" * 60)
        print("🔬 COMPARING OPTIMIZATION METHODS")
        print("=" * 60)

        # Create user-item matrix
        user_item_matrix = self.create_user_item_matrix(ratings)

        # Create train-test split
        test_ratings = []
        train_ratings = []

        for i in tqdm(range(user_item_matrix.shape[0]), desc="Processing users"):
            for j in tqdm(
                range(user_item_matrix.shape[1]), desc="Processing items", leave=False
            ):
                if user_item_matrix.iloc[i, j] > 0:
                    if np.random.random() < test_size:
                        test_ratings.append(
                            {
                                "user_idx": i,
                                "item_idx": j,
                                "rating": user_item_matrix.iloc[i, j],
                            }
                        )
                    else:
                        train_ratings.append(
                            {
                                "user_idx": i,
                                "item_idx": j,
                                "rating": user_item_matrix.iloc[i, j],
                            }
                        )

        test_df = pd.DataFrame(test_ratings)

        # Test all methods
        methods = [
            ("SGD", self.sgd_optimization),
            ("SGD + Momentum", self.sgd_momentum_optimization),
            ("Adaptive SGD", self.adaptive_sgd_optimization),
            ("ALS", self.als_optimization),
        ]

        results = {}

        for method_name, method_func in tqdm(methods, desc="Testing methods"):
            print(f"\n🧪 Testing {method_name}...")

            # Train model
            U, V = method_func(user_item_matrix.copy())

            # Evaluate
            metrics = self.evaluate_method(U, V, test_df, method_name)
            results[method_name] = {
                "metrics": metrics,
                "training_time": self.history[
                    method_name.lower().replace(" + ", "_").replace(" ", "_")
                ]["time"],
                "final_loss": self.history[
                    method_name.lower().replace(" + ", "_").replace(" ", "_")
                ]["losses"][-1],
            }

            print(f"   RMSE: {metrics['rmse']:.4f}")
            print(f"   MAE: {metrics['mae']:.4f}")
            print(f"   Training Time: {results[method_name]['training_time']:.2f}s")

        return results

    def plot_comparison(self, results):
        """
        Plot comparison of optimization methods
        """
        methods = list(results.keys())
        rmse_scores = [results[m]["metrics"]["rmse"] for m in methods]
        training_times = [results[m]["training_time"] for m in methods]
        final_losses = [results[m]["final_loss"] for m in methods]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # RMSE comparison
        axes[0].bar(
            methods, rmse_scores, color=["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"]
        )
        axes[0].set_title("RMSE Comparison")
        axes[0].set_ylabel("RMSE")
        axes[0].tick_params(axis="x", rotation=45)

        # Training time comparison
        axes[1].bar(
            methods, training_times, color=["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"]
        )
        axes[1].set_title("Training Time Comparison")
        axes[1].set_ylabel("Time (seconds)")
        axes[1].tick_params(axis="x", rotation=45)

        # Final loss comparison
        axes[2].bar(
            methods, final_losses, color=["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"]
        )
        axes[2].set_title("Final Loss Comparison")
        axes[2].set_ylabel("Loss")
        axes[2].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.show()

        # Print summary
        print("\n" + "=" * 60)
        print("📊 OPTIMIZATION METHODS SUMMARY")
        print("=" * 60)

        for method in methods:
            print(f"\n🔹 {method}:")
            print(f"   RMSE: {results[method]['metrics']['rmse']:.4f}")
            print(f"   MAE: {results[method]['metrics']['mae']:.4f}")
            print(f"   Training Time: {results[method]['training_time']:.2f}s")
            print(f"   Final Loss: {results[method]['final_loss']:.4f}")


def run_optimization_comparison():
    """
    Run comparison of all optimization methods
    """
    print("🚀 OPTIMIZATION METHODS COMPARISON")
    print("=" * 60)

    # Load data
    print("Loading MovieLens data...")
    ratings = pd.read_csv(
        "../../01_fundamentals/01_movielens_exploration/ml-1m/ratings.dat",
        sep="::",
        names=["userId", "movieId", "rating", "timestamp"],
        engine="python",
        encoding="latin-1",
    )

    # Use a subset for faster testing
    print(f"Original dataset: {len(ratings)} ratings")
    ratings_subset = ratings.sample(n=min(50000, len(ratings)), random_state=42)
    print(f"Using subset: {len(ratings_subset)} ratings")

    # Initialize optimizer
    optimizer = OptimizationMethods(
        n_factors=20,  # Reduced for faster training
        learning_rate=0.01,
        n_epochs=50,  # Reduced for faster comparison
    )

    # Run comparison
    results = optimizer.compare_optimization_methods(ratings_subset, test_size=0.2)

    # Plot results
    optimizer.plot_comparison(results)

    print(f"\n✅ Optimization methods comparison completed!")
    print(f"🎯 Key insights:")
    print(f"   - Different optimizers have different strengths")
    print(f"   - Trade-offs between speed and accuracy")
    print(f"   - ALS often provides good balance")
    print(f"   - Adaptive methods can be faster for complex data")


if __name__ == "__main__":
    run_optimization_comparison()

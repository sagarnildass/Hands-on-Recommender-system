import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")


def load_and_prepare_data():
    """
    Load MovieLens data and prepare for advanced matrix factorization
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
        index="userId",
        columns="movieId",
        values="rating",
        fill_value=0,  # Use 0 for missing values (different from NaN)
    )

    print(f"User-item matrix shape: {user_item_matrix.shape}")
    print(f"Non-zero ratings: {user_item_matrix.astype(bool).sum().sum()}")
    print(
        f"Sparsity: {(1 - user_item_matrix.astype(bool).sum().sum() / (user_item_matrix.shape[0] * user_item_matrix.shape[1])) * 100:.2f}%"
    )

    return user_item_matrix, movies, ratings


def implement_nmf(user_item_matrix, n_components=20, max_iter=200):
    """
    Implement Non-negative Matrix Factorization (NMF)
    """
    print(f"Training NMF with {n_components} components...")

    # Convert to numpy array
    R = user_item_matrix.values

    # Initialize NMF model
    nmf = NMF(
        n_components=n_components,
        max_iter=max_iter,
        random_state=42,
        alpha=0.1,  # Regularization parameter
        l1_ratio=0.5,
    )  # L1/L2 regularization ratio

    # Fit the model
    nmf.fit(R)

    # Get factor matrices
    W = nmf.transform(R)  # User factors (users × components)
    H = nmf.components_  # Item factors (components × items)

    # Reconstruct the matrix
    R_reconstructed = np.dot(W, H)

    print(f"NMF training completed!")
    print(f"User factors shape: {W.shape}")
    print(f"Item factors shape: {H.shape}")
    print(f"Reconstruction error: {nmf.reconstruction_err_:.4f}")

    return nmf, W, H, R_reconstructed


def implement_svd_plus_plus(
    user_item_matrix, n_factors=20, learning_rate=0.01, n_epochs=20, reg_param=0.1
):
    """
    Implement SVD++ (SVD with implicit feedback) - OPTIMIZED VERSION
    """
    print(f"Training SVD++ with {n_factors} factors...")

    # Convert to sparse matrix
    R = csr_matrix(user_item_matrix.values)
    n_users, n_items = R.shape

    # Initialize parameters
    U = np.random.normal(0, 0.1, (n_users, n_factors))  # User factors
    V = np.random.normal(0, 0.1, (n_items, n_factors))  # Item factors
    Y = np.random.normal(0, 0.1, (n_items, n_factors))  # Implicit feedback factors
    bu = np.zeros(n_users)  # User biases
    bi = np.zeros(n_items)  # Item biases
    mu = R.data.mean() if R.data.size > 0 else 0  # Global mean

    # Pre-compute user implicit feedback (this is the key optimization!)
    user_implicit_feedback = {}
    for user_idx in range(n_users):
        user_rated_items = R[user_idx].nonzero()[1]
        if len(user_rated_items) > 0:
            user_implicit_feedback[user_idx] = np.mean(Y[user_rated_items], axis=0)
        else:
            user_implicit_feedback[user_idx] = np.zeros(n_factors)

    # Get non-zero ratings
    user_indices, item_indices = R.nonzero()
    ratings = R.data

    print(f"Training on {len(ratings)} ratings...")

    # Training loop
    for epoch in tqdm(range(n_epochs), desc="Training"):
        total_error = 0

        # Process ratings in batches for better performance
        batch_size = 10000
        for batch_start in tqdm(
            range(0, len(ratings), batch_size), desc="Processing batches", leave=False
        ):
            batch_end = min(batch_start + batch_size, len(ratings))
            batch_user_indices = user_indices[batch_start:batch_end]
            batch_item_indices = item_indices[batch_start:batch_end]
            batch_ratings = ratings[batch_start:batch_end]

            for idx in tqdm(
                range(len(batch_ratings)), desc="Processing ratings", leave=False
            ):
                user_idx = batch_user_indices[idx]
                item_idx = batch_item_indices[idx]
                actual_rating = batch_ratings[idx]

                # Get pre-computed implicit feedback
                implicit_feedback = user_implicit_feedback[user_idx]

                # Calculate prediction
                prediction = (
                    mu
                    + bu[user_idx]
                    + bi[item_idx]
                    + np.dot(U[user_idx] + implicit_feedback, V[item_idx])
                )

                # Calculate error
                error = actual_rating - prediction
                total_error += error**2

                # Update parameters (simplified gradient descent)
                U[user_idx] += learning_rate * (
                    error * V[item_idx] - reg_param * U[user_idx]
                )
                V[item_idx] += learning_rate * (
                    error * (U[user_idx] + implicit_feedback) - reg_param * V[item_idx]
                )

                # Update biases
                bu[user_idx] += learning_rate * (error - reg_param * bu[user_idx])
                bi[item_idx] += learning_rate * (error - reg_param * bi[item_idx])

        # Print progress
        if (epoch + 1) % 5 == 0:
            rmse = np.sqrt(total_error / len(ratings))
            print(f"Epoch {epoch + 1}/{n_epochs}, RMSE: {rmse:.4f}")

    print("SVD++ training completed!")
    return U, V, Y, bu, bi, mu


def evaluate_factorization_methods(user_item_matrix, movies, ratings):
    """
    Compare different matrix factorization methods
    """
    print("\n" + "=" * 60)
    print(" COMPARING ADVANCED MATRIX FACTORIZATION METHODS")
    print("=" * 60)

    # Create train-test split
    from sklearn.model_selection import train_test_split

    # Get non-zero ratings for evaluation
    user_indices, item_indices = user_item_matrix.values.nonzero()
    rating_values = user_item_matrix.values[user_indices, item_indices]

    # Split indices
    train_indices, test_indices = train_test_split(
        range(len(rating_values)), test_size=0.2, random_state=42
    )

    # Create train and test matrices
    train_matrix = user_item_matrix.copy()
    test_matrix = user_item_matrix.copy()

    # Zero out test ratings in train matrix
    for idx in test_indices:
        train_matrix.iloc[user_indices[idx], item_indices[idx]] = 0

    # Zero out train ratings in test matrix
    for idx in train_indices:
        test_matrix.iloc[user_indices[idx], item_indices[idx]] = 0

    print("Training and evaluating methods...")

    # Test NMF
    print("\n📊 Testing NMF...")
    nmf_model, W_nmf, H_nmf, R_nmf = implement_nmf(train_matrix, n_components=20)

    # Test SVD++
    print("\n📊 Testing SVD++...")
    U_svd, V_svd, Y_svd, bu_svd, bi_svd, mu_svd = implement_svd_plus_plus(
        train_matrix,
        n_factors=10,
        learning_rate=0.01,
        n_epochs=10,  # Reduced parameters
    )

    # Evaluate on test set
    print("\n Evaluating on test set...")

    # Calculate RMSE for each method
    nmf_rmse = calculate_test_rmse(test_matrix, W_nmf, H_nmf, method="nmf")
    svd_rmse = calculate_test_rmse(
        test_matrix,
        U_svd,
        V_svd,
        method="svd_plus_plus",
        Y=Y_svd,
        bu=bu_svd,
        bi=bi_svd,
        mu=mu_svd,
    )

    print(f"\n🏆 Results:")
    print(f"NMF RMSE: {nmf_rmse:.4f}")
    print(f"SVD++ RMSE: {svd_rmse:.4f}")

    # Compare with baseline
    baseline_rmse = calculate_baseline_rmse(test_matrix)
    print(f"Baseline (mean rating) RMSE: {baseline_rmse:.4f}")

    return {"NMF": nmf_rmse, "SVD++": svd_rmse, "Baseline": baseline_rmse}


def calculate_test_rmse(
    test_matrix, user_factors, item_factors, method="nmf", **kwargs
):
    """
    Calculate RMSE on test set
    """
    test_user_indices, test_item_indices = test_matrix.values.nonzero()
    test_ratings = test_matrix.values[test_user_indices, test_item_indices]

    predictions = []

    for user_idx, item_idx in tqdm(
        zip(test_user_indices, test_item_indices),
        desc="Processing test ratings",
        leave=False,
    ):
        if method == "nmf":
            pred = np.dot(user_factors[user_idx], item_factors[:, item_idx])
        elif method == "svd_plus_plus":
            Y = kwargs.get("Y", None)
            bu = kwargs.get("bu", None)
            bi = kwargs.get("bi", None)
            mu = kwargs.get("mu", 0)

            if Y is not None and bu is not None and bi is not None:
                # Get user's rated items for implicit feedback
                user_rated_items = test_matrix.values[user_idx].nonzero()[0]
                if len(user_rated_items) > 0:
                    implicit_feedback = np.mean(Y[user_rated_items], axis=0)
                    pred = (
                        mu
                        + bu[user_idx]
                        + bi[item_idx]
                        + np.dot(
                            user_factors[user_idx] + implicit_feedback,
                            item_factors[item_idx],
                        )
                    )
                else:
                    pred = (
                        mu
                        + bu[user_idx]
                        + bi[item_idx]
                        + np.dot(user_factors[user_idx], item_factors[item_idx])
                    )
            else:
                pred = np.dot(user_factors[user_idx], item_factors[item_idx])

        predictions.append(pred)

    return np.sqrt(mean_squared_error(test_ratings, predictions))


def calculate_baseline_rmse(test_matrix):
    """
    Calculate baseline RMSE using mean rating
    """
    test_user_indices, test_item_indices = test_matrix.values.nonzero()
    test_ratings = test_matrix.values[test_user_indices, test_item_indices]

    # Use global mean as baseline
    global_mean = test_matrix.values[test_matrix.values > 0].mean()
    baseline_predictions = [global_mean] * len(test_ratings)

    return np.sqrt(mean_squared_error(test_ratings, baseline_predictions))


if __name__ == "__main__":
    # Load data
    print("Loading MovieLens data...")
    user_item_matrix, movies, ratings = load_and_prepare_data()

    # Evaluate factorization methods
    results = evaluate_factorization_methods(user_item_matrix, movies, ratings)

    # Plot results
    plt.figure(figsize=(10, 6))
    methods = list(results.keys())
    rmses = list(results.values())

    plt.bar(methods, rmses, color=["#FF6B6B", "#4ECDC4", "#45B7D1"])
    plt.title("Matrix Factorization Methods Comparison")
    plt.ylabel("RMSE (Lower is Better)")
    plt.ylim(0, max(rmses) * 1.1)

    # Add value labels on bars
    for i, v in enumerate(rmses):
        plt.text(i, v + 0.01, f"{v:.4f}", ha="center", va="bottom")

    plt.tight_layout()
    plt.show()

    print(f"\n✅ Advanced matrix factorization evaluation completed!")
    print(
        f"🏆 Best method: {min(results, key=results.get)} (RMSE: {min(results.values()):.4f})"
    )

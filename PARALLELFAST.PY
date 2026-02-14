import numpy as np
import time
import multiprocessing as mp
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ------------------------
# Helper Functions
# ------------------------

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def accuracy(X, y, w):
    preds = sigmoid(X @ w) >= 0.5
    return np.mean(preds == y)


# ------------------------
# Sequential Mini-Batch Training
# ------------------------

def train_sequential(X, y, lr=0.01, epochs=20, batch_size=1024):
    m, n = X.shape
    w = np.zeros(n)

    start = time.time()

    for _ in range(epochs):
        idx = np.random.permutation(m)
        X = X[idx]
        y = y[idx]

        for i in range(0, m, batch_size):
            Xb = X[i:i+batch_size]
            yb = y[i:i+batch_size]

            preds = sigmoid(Xb @ w)
            gradient = (Xb.T @ (preds - yb)) / len(yb)
            w -= lr * gradient

    end = time.time()

    return w, end - start


# ------------------------
# Parallel Worker
# ------------------------

def compute_gradient(args):
    X_chunk, y_chunk, w = args
    preds = sigmoid(X_chunk @ w)
    grad = (X_chunk.T @ (preds - y_chunk)) / len(y_chunk)
    return grad


# ------------------------
# Parallel Mini-Batch Training (Improved)
# ------------------------

def train_parallel(X, y, workers=4, lr=0.01, epochs=20, batch_size=1024):

    m, n = X.shape
    w = np.zeros(n)

    start = time.time()

    pool = mp.Pool(workers)

    for _ in range(epochs):
        idx = np.random.permutation(m)
        X = X[idx]
        y = y[idx]

        for i in range(0, m, batch_size):
            Xb = X[i:i+batch_size]
            yb = y[i:i+batch_size]

            chunks_X = np.array_split(Xb, workers)
            chunks_y = np.array_split(yb, workers)

            args = [(chunks_X[j], chunks_y[j], w) for j in range(workers)]

            grads = pool.map(compute_gradient, args)

            w -= lr * np.mean(grads, axis=0)

    pool.close()
    pool.join()

    end = time.time()

    return w, end - start


# ------------------------
# Main Execution
# ------------------------

def main():

    print("Generating dataset...")

    # Bigger dataset so parallel shows benefit
    X, y = make_classification(
        n_samples=100000,
        n_features=30,
        n_informative=20,
        n_classes=2,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ---------------- Sequential ----------------
    print("\nTraining sequential model...")
    w_seq, time_seq = train_sequential(X_train, y_train)
    acc_seq = accuracy(X_test, y_test, w_seq)

    print(f"Sequential Time: {time_seq:.2f} sec")
    print(f"Sequential Accuracy: {acc_seq:.4f}")

    # ---------------- Parallel ----------------
    cores_list = [1, 2, 4, 8]
    times = []
    speedups = []

    for c in cores_list:
        print(f"\nTraining parallel model with {c} cores...")
        w_par, t = train_parallel(X_train, y_train, workers=c)
        acc = accuracy(X_test, y_test, w_par)

        speedup = time_seq / t

        times.append(t)
        speedups.append(speedup)

        print(f"Cores={c}")
        print(f"Time={t:.2f} sec")
        print(f"Accuracy={acc:.4f}")
        print(f"Speedup={speedup:.2f}")

    # ---------------- Save Graphs ----------------

    plt.figure()
    plt.plot(cores_list, times, marker='o')
    plt.xlabel("Number of Cores")
    plt.ylabel("Training Time (seconds)")
    plt.title("Time vs Cores")
    plt.grid()
    plt.savefig("time_vs_cores.png", dpi=300)
    print("\ntime_vs_cores.png saved")

    plt.figure()
    plt.plot(cores_list, speedups, marker='o')
    plt.xlabel("Number of Cores")
    plt.ylabel("Speedup")
    plt.title("Speedup vs Cores")
    plt.grid()
    plt.savefig("speedup_vs_cores.png", dpi=300)
    print("speedup_vs_cores.png saved")


if __name__ == "__main__":
    mp.freeze_support()  # Required for Windows
    main()

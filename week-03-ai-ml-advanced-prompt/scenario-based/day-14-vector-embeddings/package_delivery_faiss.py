# Scenario: Package Delivery in a City
# Uses FAISS Flat and IVF indexes to find nearest warehouse to customer

import faiss
import numpy as np

print("=" * 60)
print("PACKAGE DELIVERY - FAISS NEAREST WAREHOUSE FINDER")
print("=" * 60)

d = 32        # embedding dimension
n = 2_000     # number of houses

print(f"\nCity Setup:")
print(f"  Houses      : {n:,}")
print(f"  Dimensions  : {d}")

np.random.seed(42)
vectors = np.random.randn(n, d).astype("float32")
faiss.normalize_L2(vectors)

query = np.random.randn(1, d).astype("float32")
faiss.normalize_L2(query)

# ── MODEL 1: Flat Index (Brute Force baseline) ───────────────
print("\n" + "=" * 60)
print("MODEL 1: FLAT INDEX (Brute Force)")
print("=" * 60)

index_flat = faiss.IndexFlatIP(d)
index_flat.add(vectors)
D_flat, I_flat = index_flat.search(query, k=5)

print(f"  Total vectors : {index_flat.ntotal:,}")
print(f"  Top-5 Houses  : {I_flat[0].tolist()}")
print(f"  Scores        : {[round(float(x), 4) for x in D_flat[0]]}")
print(f"  Nearest House : #{I_flat[0][0]} (score: {D_flat[0][0]:.4f})")

# ── MODEL 2: IVF Index (Neighborhood-based) ──────────────────
print("\n" + "=" * 60)
print("MODEL 2: IVF INDEX (City Neighborhoods)")
print("=" * 60)

nlist = 20
print(f"  Neighborhoods : {nlist} Voronoi cells")
print(f"  nprobe        : 3 cells searched per query")

quantizer = faiss.IndexFlatIP(d)
index_ivf = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
index_ivf.train(vectors)
index_ivf.add(vectors)
index_ivf.nprobe = 3

D_ivf, I_ivf = index_ivf.search(query, k=5)

print(f"  Total vectors : {index_ivf.ntotal:,}")
print(f"  Top-5 Houses  : {I_ivf[0].tolist()}")
print(f"  Scores        : {[round(float(x), 4) for x in D_ivf[0]]}")
print(f"  Nearest House : #{I_ivf[0][0]} (score: {D_ivf[0][0]:.4f})")

# ── COMPARISON ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("COMPARISON SUMMARY")
print("=" * 60)
print(f"\n{'Metric':<25} | {'Flat (Brute)':>14} | {'IVF':>14}")
print("-" * 58)
print(f"{'Index Type':<25} | {'Exhaustive':>14} | {'Cluster-based':>14}")
print(f"{'Top-1 Result':<25} | {f'House #{I_flat[0][0]}':>14} | {f'House #{I_ivf[0][0]}':>14}")
print(f"{'Scalability':<25} | {'Low':>14} | {'High':>14}")
print(f"{'Best for':<25} | {'Small datasets':>14} | {'Large scale':>14}")

print("\n" + "=" * 60)
print("Key Takeaways:")
print("  HNSW  -> Zoom in layer by layer (highway to street)")
print("  IVF   -> Split city into neighborhoods, search locally")
print("  Both  -> Find nearest match WITHOUT scanning all houses!")
print("=" * 60)

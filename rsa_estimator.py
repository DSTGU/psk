#!/usr/bin/env python3
"""
RSA Estimation Model Evaluator

Unix-philosophy module for evaluating RSA resource allocation models
with configurable graph topologies, models, and multiple evaluation runs.
"""

import os
import glob
import argparse
import sys
from typing import Dict, List, Tuple
from dataclasses import dataclass

import pandas as pd
import numpy as np
import networkx as nx

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.multioutput import MultiOutputRegressor

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


# -------------------------------------------------------------
# DATA STRUCTURES
# -------------------------------------------------------------
@dataclass
class RunResult:
    """Results from a single train/test run"""
    run_id: int
    mae: np.ndarray
    r2: np.ndarray
    smape: np.ndarray
    target_names: List[str]


@dataclass
class AggregateResults:
    """Aggregated statistics across multiple runs"""
    mae_mean: np.ndarray
    mae_std: np.ndarray
    mae_stderr: np.ndarray
    r2_mean: np.ndarray
    r2_std: np.ndarray
    r2_stderr: np.ndarray
    smape_mean: np.ndarray
    smape_std: np.ndarray
    smape_stderr: np.ndarray
    target_names: List[str]
    num_runs: int


# -------------------------------------------------------------
# GRAPH UTILITIES
# -------------------------------------------------------------
def load_graph(path: str) -> nx.Graph:
    """Load network topology from file"""
    with open(path) as f:
        lines = f.read().strip().splitlines()

    n = int(lines[0])
    mat = np.loadtxt(lines[2:], dtype=float)

    G = nx.Graph()
    for i in range(n):
        for j in range(i + 1, n):
            w = mat[i, j]
            if w > 0:
                G.add_edge(i, j, weight=w)

    return G


def precompute_graph_stats(G: nx.Graph) -> Dict:
    """Precompute centrality metrics for efficiency"""
    return {
        "degree": dict(G.degree()),
        "betweenness": nx.betweenness_centrality(G, weight="weight"),
        "closeness": nx.closeness_centrality(G, distance="weight"),
    }


def extract_graph_request_features(
    df: pd.DataFrame,
    G: nx.Graph,
    gstats: Dict,
) -> Dict:
    """Extract graph-topology-aware features from requests"""
    path_lengths = []
    hop_counts = []
    src_deg = []
    dst_deg = []
    src_bet = []
    dst_bet = []
    src_close = []
    dst_close = []
    max_edge_weights = []
    min_edge_weights = []
    avg_edge_weights = []

    for _, r in df.iterrows():
        s, d = int(r.source), int(r.destination)

        try:
            path = nx.shortest_path(G, s, d, weight="weight")
            length = nx.shortest_path_length(G, s, d, weight="weight")

            path_lengths.append(length)
            hop_counts.append(len(path) - 1)

            src_deg.append(gstats["degree"][s])
            dst_deg.append(gstats["degree"][d])
            src_bet.append(gstats["betweenness"][s])
            dst_bet.append(gstats["betweenness"][d])
            src_close.append(gstats["closeness"][s])
            dst_close.append(gstats["closeness"][d])

            edge_weights = [
                G[path[i]][path[i + 1]]["weight"]
                for i in range(len(path) - 1)
            ]
            
            if edge_weights:
                max_edge_weights.append(max(edge_weights))
                min_edge_weights.append(min(edge_weights))
                avg_edge_weights.append(np.mean(edge_weights))

        except nx.NetworkXNoPath:
            continue

    if not path_lengths:
        return {}

    return {
        "spath_len_mean": np.mean(path_lengths),
        "spath_len_std": np.std(path_lengths),
        "spath_len_min": np.min(path_lengths),
        "spath_len_max": np.max(path_lengths),
        "hop_count_mean": np.mean(hop_counts),
        "hop_count_std": np.std(hop_counts),
        "hop_count_max": np.max(hop_counts),
        "src_degree_mean": np.mean(src_deg),
        "src_degree_std": np.std(src_deg),
        "dst_degree_mean": np.mean(dst_deg),
        "dst_degree_std": np.std(dst_deg),
        "src_betweenness_mean": np.mean(src_bet),
        "src_betweenness_std": np.std(src_bet),
        "dst_betweenness_mean": np.mean(dst_bet),
        "dst_betweenness_std": np.std(dst_bet),
        "src_closeness_mean": np.mean(src_close),
        "dst_closeness_mean": np.mean(dst_close),
        "max_edge_weight_on_path_mean": np.mean(max_edge_weights),
        "min_edge_weight_on_path_mean": np.mean(min_edge_weights),
        "avg_edge_weight_on_path_mean": np.mean(avg_edge_weights),
    }


# -------------------------------------------------------------
# FEATURE EXTRACTION
# -------------------------------------------------------------
def extract_basic_features(df: pd.DataFrame) -> pd.Series:
    """Extract basic statistical features from request dataset"""
    feats = {
        "n_requests": len(df),
        "bitrate_sum": df["bitrate"].sum(),
        "bitrate_mean": df["bitrate"].mean(),
        "bitrate_std": df["bitrate"].std(),
        "bitrate_min": df["bitrate"].min(),
        "bitrate_max": df["bitrate"].max(),
    }
    return pd.Series(feats)


def extract_topology_features(df: pd.DataFrame) -> pd.Series:
    """Extract network topology features"""
    feats = {
        "n_unique_sources": df["source"].nunique(),
        "n_unique_destinations": df["destination"].nunique(),
        "n_unique_pairs": df[["source", "destination"]].drop_duplicates().shape[0],
        "mean_requests_per_source": df["source"].value_counts().mean(),
        "std_requests_per_source": df["source"].value_counts().std(),
        "mean_requests_per_destination": df["destination"].value_counts().mean(),
        "std_requests_per_destination": df["destination"].value_counts().std(),
        "max_requests_per_source": df["source"].value_counts().max(),
        "max_requests_per_destination": df["destination"].value_counts().max(),
    }
    return pd.Series(feats)


def extract_distribution_features(df: pd.DataFrame) -> pd.Series:
    """Extract distribution and percentile features"""
    feats = {
        "bitrate_q25": df["bitrate"].quantile(0.25),
        "bitrate_q50": df["bitrate"].quantile(0.50),
        "bitrate_q75": df["bitrate"].quantile(0.75),
        "bitrate_q90": df["bitrate"].quantile(0.90),
        "bitrate_q95": df["bitrate"].quantile(0.95),
        "bitrate_iqr": df["bitrate"].quantile(0.75) - df["bitrate"].quantile(0.25),
        "bitrate_skew": df["bitrate"].skew(),
        "bitrate_kurtosis": df["bitrate"].kurtosis(),
        "bitrate_cv": df["bitrate"].std() / df["bitrate"].mean() if df["bitrate"].mean() > 0 else 0,
    }
    return pd.Series(feats)


def extract_traffic_features(df: pd.DataFrame) -> pd.Series:
    """Extract traffic pattern features"""
    # Aggregate bitrate per source and destination
    src_bitrate = df.groupby("source")["bitrate"].sum()
    dst_bitrate = df.groupby("destination")["bitrate"].sum()
    
    # Pair-wise traffic
    pair_traffic = df.groupby(["source", "destination"])["bitrate"].agg(["sum", "count"])
    
    feats = {
        "traffic_per_source_mean": src_bitrate.mean(),
        "traffic_per_source_std": src_bitrate.std(),
        "traffic_per_source_max": src_bitrate.max(),
        "traffic_per_destination_mean": dst_bitrate.mean(),
        "traffic_per_destination_std": dst_bitrate.std(),
        "traffic_per_destination_max": dst_bitrate.max(),
        "traffic_per_pair_mean": pair_traffic["sum"].mean(),
        "traffic_per_pair_std": pair_traffic["sum"].std(),
        "traffic_per_pair_max": pair_traffic["sum"].max(),
        "requests_per_pair_mean": pair_traffic["count"].mean(),
        "requests_per_pair_max": pair_traffic["count"].max(),
    }
    return pd.Series(feats)


def extract_features(df: pd.DataFrame, feature_set: str) -> pd.Series:
    """Extract features based on selected feature set"""
    all_feats = pd.Series(dtype=float)
    
    if feature_set in ["basic", "all"]:
        all_feats = pd.concat([all_feats, extract_basic_features(df)])
    
    if feature_set in ["topology", "all"]:
        all_feats = pd.concat([all_feats, extract_topology_features(df)])
    
    if feature_set in ["distribution", "all"]:
        all_feats = pd.concat([all_feats, extract_distribution_features(df)])
    
    if feature_set in ["traffic", "all"]:
        all_feats = pd.concat([all_feats, extract_traffic_features(df)])
    
    return all_feats


# -------------------------------------------------------------
# DATA LOADING
# -------------------------------------------------------------
def load_topology(
    path: str,
    topology_name: str,
    graph_file: str,
    feature_set: str = "all",
    use_graph_features: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    """Load all request sets for a given topology"""
    G = None
    gstats = None
    
    if use_graph_features:
        G = load_graph(graph_file)
        gstats = precompute_graph_stats(G)

    rows = []
    request_sets = sorted(glob.glob(os.path.join(path, "request-set_*")))

    for rs in request_sets:
        req_path = os.path.join(rs, "requests.csv")
        res_path = os.path.join(rs, "results.txt")

        if not (os.path.exists(req_path) and os.path.exists(res_path)):
            continue

        df_req = pd.read_csv(req_path)

        # Parse results
        metrics = {}
        with open(res_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) < 2:
                    continue

                key = parts[0]
                val_str = parts[-1].replace(",", ".")
                try:
                    value = float(val_str)
                except ValueError:
                    continue

                metrics[key] = value

        # Extract request-based features
        feat = extract_features(df_req, feature_set)

        # Extract graph-based features if enabled
        if use_graph_features and G is not None:
            graph_feats = extract_graph_request_features(df_req, G, gstats)
            for k, v in graph_feats.items():
                feat[k] = v

        target_keys = [
            "highestSlot",
            "avgHighestSlot",
            "sumOfSlots",
            "avgActiveTransceivers",
        ]
        for key in target_keys:
            feat[key] = metrics.get(key, None)

        feat["topology"] = topology_name
        feat["request_set"] = os.path.basename(rs)

        rows.append(feat)

    if verbose:
        print(f"Loaded {len(rows)} request sets from {topology_name}", file=sys.stderr)

    return pd.DataFrame(rows)


def load_data(
    graph: str,
    feature_set: str = "all",
    use_graph_features: bool = True,
    base_path: str = "RSA_estimation",
    verbose: bool = False,
) -> pd.DataFrame:
    """Load data based on graph selection"""
    data_frames = []

    if graph in ["euro28", "both"]:
        df_euro = load_topology(
            os.path.join(base_path, "Euro28"),
            "Euro28",
            os.path.join(base_path, "Euro28", "euro28.net"),
            feature_set=feature_set,
            use_graph_features=use_graph_features,
            verbose=verbose,
        )
        data_frames.append(df_euro)

    if graph in ["us26", "both"]:
        df_us = load_topology(
            os.path.join(base_path, "US26"),
            "US26",
            os.path.join(base_path, "US26", "us26.net"),
            feature_set=feature_set,
            use_graph_features=use_graph_features,
            verbose=verbose,
        )
        data_frames.append(df_us)

    data = pd.concat(data_frames, ignore_index=True)

    target_cols = [
        "highestSlot",
        "avgHighestSlot",
        "sumOfSlots",
        "avgActiveTransceivers",
    ]

    data = data.dropna(subset=target_cols)

    if verbose:
        print(f"Total examples: {len(data)}", file=sys.stderr)
        print(f"Feature set: {feature_set}", file=sys.stderr)
        print(f"Graph features: {'enabled' if use_graph_features else 'disabled'}", file=sys.stderr)
        print(f"Total features: {len([c for c in data.columns if c not in target_cols + ['topology', 'request_set']])}", file=sys.stderr)

    return data


# -------------------------------------------------------------
# MODEL FACTORY
# -------------------------------------------------------------
def create_model(model_name: str, random_state: int = 42):
    """Factory for creating ML models"""
    if model_name == "xgboost":
        return MultiOutputRegressor(
            XGBRegressor(
                n_estimators=400,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=random_state,
            )
        )
    elif model_name == "lightgbm":
        return MultiOutputRegressor(
            LGBMRegressor(
                n_estimators=400,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=random_state,
                verbosity=-1,
            )
        )
    elif model_name == "catboost":
        return CatBoostRegressor(
            depth=6,
            learning_rate=0.05,
            n_estimators=400,
            loss_function="MultiRMSE",
            verbose=False,
            random_seed=random_state,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


# -------------------------------------------------------------
# EVALUATION
# -------------------------------------------------------------
def smape(y_true, y_pred, epsilon=1e-8):
    """
    Symmetric Mean Absolute Percentage Error (sMAPE)
    Returns per-target values.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    denom = np.abs(y_true) + np.abs(y_pred) + epsilon
    return 2.0 * np.mean(np.abs(y_pred - y_true) / denom, axis=0)


def run_single_evaluation(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    run_id: int,
    model_name: str,
) -> RunResult:
    """Perform a single train/test evaluation"""
    if model_name == "catboost":
        model.fit(X_train, y_train.values)
    else:
        model.fit(X_train, y_train)

    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds, multioutput="raw_values")
    r2 = r2_score(y_test, preds, multioutput="raw_values")
    smape_vals = smape(y_test.values, preds)

    return RunResult(
        run_id=run_id,
        mae=mae,
        r2=r2,
        smape=smape_vals,
        target_names=list(y_test.columns),
    )


def aggregate_results(results: List[RunResult]) -> AggregateResults:
    """Aggregate statistics across multiple runs"""
    mae_values = np.array([r.mae for r in results])
    r2_values = np.array([r.r2 for r in results])
    smape_values = np.array([r.smape for r in results])

    n = len(results)

    return AggregateResults(
        mae_mean=np.mean(mae_values, axis=0),
        mae_std=np.std(mae_values, axis=0, ddof=1),
        mae_stderr=np.std(mae_values, axis=0, ddof=1) / np.sqrt(n),

        r2_mean=np.mean(r2_values, axis=0),
        r2_std=np.std(r2_values, axis=0, ddof=1),
        r2_stderr=np.std(r2_values, axis=0, ddof=1) / np.sqrt(n),
        
        smape_mean=np.mean(smape_values, axis=0),
        smape_std=np.std(smape_values, axis=0, ddof=1),
        smape_stderr=np.std(smape_values, axis=0, ddof=1) / np.sqrt(n),
        
        target_names=results[0].target_names,
        num_runs=n,
    )


# -------------------------------------------------------------
# OUTPUT FORMATTING
# -------------------------------------------------------------
def print_run_result(result: RunResult, verbose: bool = False):
    """Print results from a single run"""
    if verbose:
        print(f"\n--- Run {result.run_id} ---")
        for i, name in enumerate(result.target_names):
            print(f"{name:22s} | MAE={result.mae[i]:8.3f} | R²={result.r2[i]:6.3f}")


def print_aggregate_results(
    agg: AggregateResults,
    model_name: str,
    graph: str,
    feature_set: str,
    use_graph_features: bool,
):
    """Print aggregated results with statistics"""
    print(f"\n{'='*70}")
    print(f"Model: {model_name.upper()} | Graph: {graph.upper()} | Runs: {agg.num_runs}")
    print(f"Features: {feature_set} | Graph features: {'enabled' if use_graph_features else 'disabled'}")
    print(f"{'='*70}")
    print(f"\n{'Target':<22s} | {'MAE Mean':>10s} ± {'StdDev':>8s} (±{'StdErr':>8s}) | {'R² Mean':>8s} ± {'StdDev':>8s} | {'Smape':>8s} ± {'StdDev':>8s}")
    print("-" * 110)

    for i, name in enumerate(agg.target_names):
        print(
            f"{name:<22s} | "
            f"{agg.mae_mean[i]:10.3f} ± {agg.mae_std[i]:8.3f} (±{agg.mae_stderr[i]:8.3f}) | "
            f"{agg.r2_mean[i]:8.3f} ± {agg.r2_std[i]:8.3f} | "
            f"{agg.smape_mean[i]:8.3f} ± {agg.smape_std[i]:8.3f}"
        )

    print("\nMeasurement Uncertainty (95% CI, ±1.96σ):")
    for i, name in enumerate(agg.target_names):
        ci95 = 1.96 * agg.mae_stderr[i]
        print(f"  {name:<22s} | MAE: {agg.mae_mean[i]:.3f} ± {ci95:.3f}")


# -------------------------------------------------------------
# MAIN
# -------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RSA estimation models on network topologies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --graph euro28 --model xgboost --runs 10
  %(prog)s --graph both --model catboost --runs 5 --verbose
  %(prog)s --graph us26 --model lightgbm --runs 1 --seed 123
  %(prog)s --graph both --model xgboost --features basic --no-graph-features
  %(prog)s --graph both --model catboost --features traffic --runs 10

Feature sets:
  basic        - Request statistics (count, bitrate stats)
  topology     - Network topology usage (sources, destinations, pairs)
  distribution - Statistical distributions (quantiles, skewness, kurtosis)
  traffic      - Traffic patterns (per-source, per-destination, per-pair)
  all          - All of the above (default)
        """,
    )

    parser.add_argument(
        "--graph",
        choices=["euro28", "us26", "both"],
        required=True,
        help="Network topology to evaluate",
    )

    parser.add_argument(
        "--model",
        choices=["catboost", "lightgbm", "xgboost"],
        required=True,
        help="ML model to use",
    )

    parser.add_argument(
        "--features",
        choices=["basic", "topology", "distribution", "traffic", "all"],
        default="all",
        help="Feature set to extract (default: all)",
    )

    parser.add_argument(
        "--no-graph-features",
        action="store_true",
        help="Disable graph-based features (shortest paths, centrality)",
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of train/test cycles (default: 5)",
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set proportion (default: 0.2)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    parser.add_argument(
        "--base-path",
        type=str,
        default="RSA_estimation",
        help="Base path to data directory (default: RSA_estimation)",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show individual run results",
    )

    args = parser.parse_args()

    # Load data
    print(f"Loading data for graph: {args.graph}...", file=sys.stderr)
    data = load_data(
        args.graph,
        feature_set=args.features,
        use_graph_features=not args.no_graph_features,
        base_path=args.base_path,
        verbose=args.verbose,
    )

    target_cols = [
        "highestSlot",
        "avgHighestSlot",
        "sumOfSlots",
        "avgActiveTransceivers",
    ]

    feature_cols = [
        c for c in data.columns
        if c not in target_cols + ["topology", "request_set"]
    ]

    X = data[feature_cols]
    y = data[target_cols]

    if args.verbose:
        print(f"\nFeature columns ({len(feature_cols)}):", file=sys.stderr)
        for fc in sorted(feature_cols):
            print(f"  - {fc}", file=sys.stderr)
        print(file=sys.stderr)

    # Run evaluations
    results = []
    for run_id in range(1, args.runs + 1):
        # Use different random state for each run
        run_seed = args.seed + run_id

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=run_seed
        )

        model = create_model(args.model, random_state=run_seed)

        result = run_single_evaluation(
            model, X_train, X_test, y_train, y_test, run_id, args.model
        )

        results.append(result)
        print_run_result(result, verbose=args.verbose)

    # Aggregate and display results
    agg = aggregate_results(results)
    print_aggregate_results(
        agg,
        args.model,
        args.graph,
        args.features,
        not args.no_graph_features,
    )


if __name__ == "__main__":
    main()

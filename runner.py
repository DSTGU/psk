#!/usr/bin/env python3
"""
RSA Estimator Batch Runner

Runs all combinations of models, graphs, and feature sets,
collecting results into a CSV file for analysis.
"""

import argparse
import subprocess
import sys
import csv
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
from itertools import product


# Configuration
MODELS = ["xgboost", "lightgbm", "catboost"]
GRAPHS = ["euro28", "us26", "both"]
FEATURE_SETS = ["basic", "topology", "distribution", "traffic", "all"]
GRAPH_FEATURE_OPTIONS = [True, False]  # With and without graph features

TARGET_METRICS = [
    "highestSlot",
    "avgHighestSlot",
    "sumOfSlots",
    "avgActiveTransceivers",
]


def parse_output(output: str) -> Dict[str, Dict[str, float]]:
    """
    Parse the output from rsa_estimator.py to extract metrics.

    Returns:
    {
        target: {
            mae_mean, mae_std, mae_stderr,
            r2_mean, r2_std
        }
    }
    """
    results: Dict[str, Dict[str, float]] = {}

    lines = output.splitlines()
    in_table = False

    # Relaxed numeric pattern (handles ints, floats, negatives)
    num = r"-?\d+(?:\.\d+)?"

    row_re = re.compile(
        rf"""
        ^\s*(?P<target>\S+)\s*\|\s*
        (?P<mae_mean>{num})\s*±\s*(?P<mae_std>{num})\s*
        \(±\s*(?P<mae_stderr>{num})\)\s*\|\s*
        (?P<r2_mean>{num})\s*±\s*(?P<r2_std>{num})
        """,
        re.VERBOSE,
    )

    for line in lines:
        # Start parsing after header row
        if "MAE Mean" in line and "|" in line:
            in_table = True
            continue

        if not in_table:
            continue

        print(line)

        # Stop at next section
        if line.strip().startswith("Measurement Uncertainty"):
            break

        # Skip separators / empty lines
        if not line.strip() or set(line.strip()) <= {"-", "="}:
            continue

        match = row_re.match(line)
        if not match:
            continue

        g = match.groupdict()
        results[g["target"]] = {
            "mae_mean": float(g["mae_mean"]),
            "mae_std": float(g["mae_std"]),
            "mae_stderr": float(g["mae_stderr"]),
            "r2_mean": float(g["r2_mean"]),
            "r2_std": float(g["r2_std"]),
        }

    return results


def run_experiment(
    model: str,
    graph: str,
    features: str,
    use_graph_features: bool,
    runs: int,
    seed: int,
    base_path: str,
    estimator_script: str = "./rsa_estimator.py",
) -> Tuple[bool, Dict[str, Dict[str, float]], str]:
    """
    Run a single experiment configuration.
    
    Returns:
        (success, results_dict, error_message)
    """
    cmd = [
        sys.executable,
        estimator_script,
        "--graph", graph,
        "--model", model,
        "--features", features,
        "--runs", str(runs),
        "--seed", str(seed),
        "--base-path", base_path,
    ]
    
    if not use_graph_features:
        cmd.append("--no-graph-features")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
        )
        
        if result.returncode != 0:
            return False, {}, f"Non-zero exit code: {result.returncode}\n{result.stderr}"
        
        parsed_results = parse_output(result.stdout)
        
        if not parsed_results:
            return False, {}, "Failed to parse output"
        
        return True, parsed_results, ""
        
    except subprocess.TimeoutExpired:
        return False, {}, "Timeout after 10 minutes"
    except Exception as e:
        return False, {}, f"Exception: {str(e)}"


def write_results_header(writer: csv.DictWriter):
    """Write CSV header"""
    writer.writeheader()


def create_result_row(
    model: str,
    graph: str,
    features: str,
    use_graph_features: bool,
    runs: int,
    target: str,
    metrics: Dict[str, float],
    status: str,
    error: str = "",
) -> Dict:
    """Create a single row for the results CSV"""
    return {
        'timestamp': datetime.now().isoformat(),
        'model': model,
        'graph': graph,
        'features': features,
        'graph_features': 'yes' if use_graph_features else 'no',
        'runs': runs,
        'target': target,
        'mae_mean': metrics.get('mae_mean', ''),
        'mae_std': metrics.get('mae_std', ''),
        'mae_stderr': metrics.get('mae_stderr', ''),
        'r2_mean': metrics.get('r2_mean', ''),
        'r2_std': metrics.get('r2_std', ''),
        'status': status,
        'error': error,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run all combinations of RSA estimator configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script runs rsa_estimator.py with all combinations of:
  - Models: xgboost, lightgbm, catboost
  - Graphs: euro28, us26, both
  - Features: basic, topology, distribution, traffic, all
  - Graph features: enabled, disabled

Examples:
  %(prog)s --runs 10 --output results.csv
  %(prog)s --runs 5 --output results.csv --models xgboost catboost
  %(prog)s --runs 10 --graphs both --features all --output quick_test.csv
        """,
    )
    
    parser.add_argument(
        "--runs",
        type=int,
        required=True,
        help="Number of runs per configuration",
    )
    
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output CSV file path",
    )
    
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODELS,
        default=MODELS,
        help=f"Models to test (default: all)",
    )
    
    parser.add_argument(
        "--graphs",
        nargs="+",
        choices=GRAPHS,
        default=GRAPHS,
        help=f"Graphs to test (default: all)",
    )
    
    parser.add_argument(
        "--features",
        nargs="+",
        choices=FEATURE_SETS,
        default=FEATURE_SETS,
        help=f"Feature sets to test (default: all)",
    )
    
    parser.add_argument(
        "--graph-features",
        choices=["yes", "no", "both"],
        default="both",
        help="Test with graph features (default: both)",
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
        "--estimator-script",
        type=str,
        default="./rsa_estimator.py",
        help="Path to rsa_estimator.py (default: ./rsa_estimator.py)",
    )
    
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running even if some experiments fail",
    )
    
    args = parser.parse_args()
    
    # Determine graph feature options to test
    if args.graph_features == "yes":
        graph_feature_options = [True]
    elif args.graph_features == "no":
        graph_feature_options = [False]
    else:  # "both"
        graph_feature_options = [True, False]
    
    # Generate all combinations
    combinations = list(product(
        args.models,
        args.graphs,
        args.features,
        graph_feature_options,
    ))
    
    total_experiments = len(combinations)
    
    print(f"{'='*70}", file=sys.stderr)
    print(f"RSA Estimator Batch Runner", file=sys.stderr)
    print(f"{'='*70}", file=sys.stderr)
    print(f"Total experiments: {total_experiments}", file=sys.stderr)
    print(f"Runs per experiment: {args.runs}", file=sys.stderr)
    print(f"Output file: {args.output}", file=sys.stderr)
    print(f"{'='*70}\n", file=sys.stderr)
    
    # Prepare CSV file
    fieldnames = [
        'timestamp',
        'model',
        'graph',
        'features',
        'graph_features',
        'runs',
        'target',
        'mae_mean',
        'mae_std',
        'mae_stderr',
        'r2_mean',
        'r2_std',
        'status',
        'error',
    ]
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        successful = 0
        failed = 0
        
        for idx, (model, graph, features, use_graph_features) in enumerate(combinations, 1):
            config_str = (
                f"{model}/{graph}/{features}/"
                f"{'with' if use_graph_features else 'no'}-graph-features"
            )
            
            print(
                f"[{idx}/{total_experiments}] Running: {config_str}...",
                file=sys.stderr,
                flush=True,
            )
            
            success, results, error = run_experiment(
                model=model,
                graph=graph,
                features=features,
                use_graph_features=use_graph_features,
                runs=args.runs,
                seed=args.seed,
                base_path=args.base_path,
                estimator_script=args.estimator_script,
            )
            
            if success:
                # Write one row per target metric
                for target in TARGET_METRICS:
                    if target in results:
                        row = create_result_row(
                            model=model,
                            graph=graph,
                            features=features,
                            use_graph_features=use_graph_features,
                            runs=args.runs,
                            target=target,
                            metrics=results[target],
                            status='success',
                        )
                        writer.writerow(row)
                
                successful += 1
                print(f"  ✓ Success", file=sys.stderr)
            else:
                # Write failure row
                row = create_result_row(
                    model=model,
                    graph=graph,
                    features=features,
                    use_graph_features=use_graph_features,
                    runs=args.runs,
                    target='',
                    metrics={},
                    status='failed',
                    error=error,
                )
                writer.writerow(row)
                
                failed += 1
                print(f"  ✗ Failed: {error}", file=sys.stderr)
                
                if not args.continue_on_error:
                    print("\nStopping due to error. Use --continue-on-error to continue.", file=sys.stderr)
                    sys.exit(1)
            
            # Flush after each experiment
            csvfile.flush()
    
    print(f"\n{'='*70}", file=sys.stderr)
    print(f"Batch run complete!", file=sys.stderr)
    print(f"Successful: {successful}/{total_experiments}", file=sys.stderr)
    print(f"Failed: {failed}/{total_experiments}", file=sys.stderr)
    print(f"Results written to: {args.output}", file=sys.stderr)
    print(f"{'='*70}", file=sys.stderr)
    
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()

"""
Evaluation utility functions
"""
from typing import Dict, Any, List
import pandas as pd


def print_evaluation_results(results, title: str = "Evaluation Results"):
    """Print formatted evaluation results"""
    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}")
    
    # Print metrics
    if hasattr(results, 'metrics') and results.metrics:
        print("\n📊 Metrics:")
        for metric_name, value in results.metrics.items():
            if isinstance(value, float):
                print(f"  {metric_name}: {value:.4f}")
            else:
                print(f"  {metric_name}: {value}")
    
    # Print tables info
    if hasattr(results, 'tables') and results.tables:
        print(f"\n📋 Tables available: {list(results.tables.keys())}")
        if 'eval_results_table' in results.tables:
            df = results.tables['eval_results_table']
            print(f"\n   Evaluation results table shape: {df.shape}")
            print(f"   Columns: {list(df.columns)}")
    
    print(f"{'='*70}\n")


def compare_results(results_dict: Dict[str, Any], metric_name: str = None):
    """Compare multiple evaluation results"""
    print(f"\n{'='*70}")
    print(f"Results Comparison")
    print(f"{'='*70}\n")
    
    if not results_dict:
        print("No results to compare")
        return
    
    # Create comparison DataFrame
    comparison_data = []
    for name, results in results_dict.items():
        row = {"name": name}
        if hasattr(results, 'metrics'):
            row.update(results.metrics)
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    print(df.to_string(index=False))
    print(f"\n{'='*70}\n")


def extract_metrics_from_results(results) -> Dict[str, float]:
    """Extract metrics dictionary from evaluation results"""
    if hasattr(results, 'metrics'):
        return results.metrics
    return {}


def print_trace_info(trace):
    """Print trace information"""
    print(f"\n{'='*70}")
    print("Trace Information")
    print(f"{'='*70}")
    print(f"Trace ID: {trace.info.trace_id}")
    print(f"Request ID: {trace.info.request_id}")
    print(f"Timestamp: {trace.info.timestamp_ms}")
    
    if hasattr(trace.info, 'token_usage') and trace.info.token_usage:
        print(f"\n🪙 Token Usage:")
        usage = trace.info.token_usage
        print(f"  Input tokens: {usage.get('input_tokens', 0)}")
        print(f"  Output tokens: {usage.get('output_tokens', 0)}")
        print(f"  Total tokens: {usage.get('total_tokens', 0)}")
    
    if hasattr(trace, 'data') and trace.data.spans:
        print(f"\n📍 Spans: {len(trace.data.spans)}")
        for i, span in enumerate(trace.data.spans[:3], 1):  # Show first 3
            print(f"  {i}. {span.name} ({span.span_type})")
    
    print(f"{'='*70}\n")


def print_scorer_info(scorer, name: str = "Scorer"):
    """Print information about a scorer"""
    print(f"\n{'='*60}")
    print(f"{name} Information")
    print(f"{'='*60}")
    print(f"Name: {getattr(scorer, 'name', 'N/A')}")
    if hasattr(scorer, 'model'):
        print(f"Model: {scorer.model}")
    if hasattr(scorer, 'instructions'):
        print(f"Instructions: {scorer.instructions[:100]}...")
    print(f"{'='*60}\n")


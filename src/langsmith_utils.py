import os
import logging
from functools import wraps
from typing import Any, Callable, Dict, Optional
from langsmith import traceable, Client
from langsmith.run_helpers import get_current_run_tree

logger = logging.getLogger(__name__)


def is_langsmith_enabled() -> bool:
    """Check if LangSmith tracing is enabled."""
    return os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"


def get_langsmith_client() -> Optional[Client]:
    """Get LangSmith client if enabled."""
    if not is_langsmith_enabled():
        return None
    
    try:
        return Client()
    except Exception as e:
        logger.warning(f"Failed to initialize LangSmith client: {e}")
        return None


def trace_node(node_name: str, node_type: str = "pipeline"):
    """
    Decorator to trace LangGraph nodes with LangSmith.
    
    Usage:
        @trace_node("ingest", "data_collection")
        def ingest_node(state: AgentState) -> AgentState:
            ...
    """
    def decorator(func: Callable) -> Callable:
        if not is_langsmith_enabled():
            return func
        
        @traceable(
            name=node_name,
            tags=[node_type, "pipeline", "node"],
            metadata={"node_type": node_type}
        )
        @wraps(func)
        def wrapper(state, *args, **kwargs):
            # Extract state metadata for tracing
            metadata = {
                "iteration": state.iteration_count,
                "max_iterations": state.max_iterations,
                "coverage_threshold": state.coverage_threshold,
                "num_targets": len(state.targets) if hasattr(state, 'targets') else 0,
                "num_source_files": len(state.source_map) if hasattr(state, 'source_map') else 0,
            }
            
            # Add coverage info if available
            if hasattr(state, 'coverage_report') and state.coverage_report:
                coverage_data = state.coverage_report.get('summary', {})
                if isinstance(coverage_data, dict):
                    inner = coverage_data.get('summary', {})
                    if inner:
                        metadata["current_coverage"] = inner.get('percent', 0)
            
            # Execute the node
            try:
                result = func(state, *args, **kwargs)
                
                # Add output metadata
                run_tree = get_current_run_tree()
                if run_tree:
                    run_tree.extra = run_tree.extra or {}
                    run_tree.extra.update(metadata)
                    run_tree.extra["status"] = "success"
                
                return result
            
            except Exception as e:
                # Log error to LangSmith
                run_tree = get_current_run_tree()
                if run_tree:
                    run_tree.extra = run_tree.extra or {}
                    run_tree.extra.update(metadata)
                    run_tree.extra["status"] = "error"
                    run_tree.extra["error_message"] = str(e)
                
                raise
        
        return wrapper
    return decorator


def log_metric(metric_name: str, value: Any, metadata: Optional[Dict] = None):
    """
    Log a custom metric to LangSmith.
    
    Args:
        metric_name: Name of the metric
        value: Metric value
        metadata: Optional additional context
    """
    if not is_langsmith_enabled():
        return
    
    try:
        run_tree = get_current_run_tree()
        if run_tree:
            run_tree.extra = run_tree.extra or {}
            run_tree.extra[metric_name] = value
            
            if metadata:
                run_tree.extra[f"{metric_name}_metadata"] = metadata
    except Exception as e:
        logger.debug(f"Failed to log metric to LangSmith: {e}")


def create_run_summary(state) -> Dict[str, Any]:
    """
    Create a comprehensive summary for LangSmith.
    
    Returns dict with key metrics from the pipeline run.
    """
    summary = {
        "iterations_completed": state.iteration_count,
        "max_iterations": state.max_iterations,
        "coverage_threshold": state.coverage_threshold,
    }
    
    # Coverage stats
    if hasattr(state, 'coverage_report') and state.coverage_report:
        coverage_data = state.coverage_report.get('summary', {})
        if isinstance(coverage_data, dict):
            inner = coverage_data.get('summary', {})
            if inner:
                summary["final_coverage"] = inner.get('percent', 0)
                summary["statements_covered"] = inner.get('covered', 0)
                summary["statements_total"] = inner.get('statements', 0)
    
    # Test stats
    if hasattr(state, 'execution_report') and state.execution_report:
        exec_summary = state.execution_report.get('summary', {})
        if exec_summary:
            summary["total_tests"] = exec_summary.get('tests', 0)
            summary["tests_passed"] = exec_summary.get('tests', 0) - exec_summary.get('failures', 0) - exec_summary.get('errors', 0)
            summary["tests_failed"] = exec_summary.get('failures', 0)
            summary["tests_errors"] = exec_summary.get('errors', 0)
    
    # Intelligence metrics
    if hasattr(state, 'coverage_patterns'):
        summary["patterns_identified"] = len(state.coverage_patterns)
    
    if hasattr(state, 'prioritized_targets'):
        summary["targets_prioritized"] = len(state.prioritized_targets)
    
    if hasattr(state, 'skipped_targets'):
        summary["targets_skipped"] = len(state.skipped_targets)
    
    # File stats
    if hasattr(state, 'source_map'):
        summary["source_files_processed"] = len(state.source_map)
    
    if hasattr(state, 'targets'):
        summary["targets_found"] = len(state.targets)
    
    if hasattr(state, 'generated_tests'):
        summary["test_files_generated"] = len(state.generated_tests)
    
    return summary
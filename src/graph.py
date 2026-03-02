import logging
from langgraph.graph import StateGraph, END
from schema import AgentState

from nodes.ingestor import ingest_node
from nodes.generator import generate_node
from nodes.writer import write_node
from nodes.dependency_installer import dependency_installer_node
from nodes.executor import executor_node
from nodes.coverage_analyzer import coverage_analyzer_node
from nodes.gap_analyzer import analyze_gaps_node
from nodes.supplement_generator import supplement_tests_node
from nodes.test_appender import append_tests_node
from nodes.finalizer import finalize_node
from nodes.pattern_classifier import pattern_classifier_node
from nodes.smart_prioritizer import smart_prioritizer_node

logger = logging.getLogger(__name__)


def should_iterate(state: AgentState) -> str:
    """Decision logic with better failure handling."""
    coverage_report = state.coverage_report
    
    # Safety: No coverage report
    if not coverage_report:
        logger.warning("=" * 60)
        logger.warning("⚠️ DECISION: FINALIZE (No coverage report)")
        logger.warning("=" * 60)
        return "finalize"
    
    # Check if NO TESTS were generated at all
    if not state.generated_tests:
        logger.error("=" * 60)
        logger.error("✗ DECISION: FINALIZE (No tests generated)")
        logger.error("=" * 60)
        return "finalize"
    
    # Extract coverage
    outer_summary = coverage_report.get('summary', {})
    if isinstance(outer_summary, dict):
        inner_summary = outer_summary.get('summary', {})
        current_coverage = inner_summary.get('percent', 0) if inner_summary else outer_summary.get('percent', 0)
    else:
        current_coverage = 0
    
    threshold = state.coverage_threshold
    
    # Check 1: Coverage threshold met
    if current_coverage >= threshold:
        logger.info("=" * 60)
        logger.info("✓ DECISION: FINALIZE (Coverage goal met)")
        logger.info(f"  Current: {current_coverage:.1f}% ≥ Threshold: {threshold}%")
        logger.info("=" * 60)
        return "finalize"
    
    # Check 2: Max iterations reached
    if state.iteration_count >= state.max_iterations:
        logger.info("=" * 60)
        logger.warning("✗ DECISION: FINALIZE (Max iterations reached)")
        logger.warning(f"  Iterations: {state.iteration_count}/{state.max_iterations}")
        logger.warning(f"  Current coverage: {current_coverage:.1f}%")
        logger.info("=" * 60)
        return "finalize"
    
    # Check 3: Test execution failed
    exec_report = state.execution_report
    if exec_report:
        returncode = exec_report.get('returncode', 0)
        if returncode in [2, 5]:
            logger.error("=" * 60)
            logger.error("✗ DECISION: FINALIZE (Test execution failed)")
            logger.error(f"  Return code: {returncode}")
            logger.error("=" * 60)
            return "finalize"
        
        summary = exec_report.get('summary', {})
        errors = summary.get('errors', 0)
        if errors > 0 and summary.get('tests', 0) == 0:
            logger.error("=" * 60)
            logger.error("✗ DECISION: FINALIZE (All tests have errors)")
            logger.error("=" * 60)
            return "finalize"
    
    # Check 4: No progress
    if state.previous_coverage is not None:
        progress = current_coverage - state.previous_coverage
        if progress <= 0.1:
            logger.info("=" * 60)
            logger.warning("✗ DECISION: FINALIZE (No progress)")
            logger.warning(f"  Improvement: {progress:.2f}%")
            logger.info("=" * 60)
            return "finalize"
    
    # Continue iterating
    logger.info("=" * 60)
    logger.info("⟳ DECISION: CONTINUE ITERATION")
    logger.info(f"  Current coverage: {current_coverage:.1f}% < Threshold: {threshold}%")
    logger.info(f"  Starting iteration: {state.iteration_count + 1}/{state.max_iterations}")
    logger.info("=" * 60)
    
    return "classify"


def build_graph():
    g = StateGraph(AgentState)
    
    # Initial generation phase
    g.add_node("ingest", ingest_node)
    g.add_node("generate", generate_node)
    g.add_node("write", write_node)
    g.add_node("resolve_deps", dependency_installer_node)
    g.add_node("execute", executor_node)
    g.add_node("coverage", coverage_analyzer_node)
    
    # Phase 1: Intelligence Layer nodes (NO effectiveness_predictor)
    g.add_node("pattern_classifier", pattern_classifier_node)
    g.add_node("smart_prioritizer", smart_prioritizer_node)
    
    # Feedback loop nodes
    g.add_node("analyze_gaps", analyze_gaps_node)
    g.add_node("supplement_tests", supplement_tests_node)
    g.add_node("append_tests", append_tests_node)
    
    # Finalization
    g.add_node("finalize", finalize_node)
    
    # Initial flow
    g.set_entry_point("ingest")
    g.add_edge("ingest", "generate")
    g.add_edge("generate", "write")
    g.add_edge("write", "resolve_deps")
    g.add_edge("resolve_deps", "execute")
    g.add_edge("execute", "coverage")
    
    # Add pattern classifier after coverage
    g.add_edge("coverage", "pattern_classifier")
    
    # Decision point
    g.add_conditional_edges(
        "pattern_classifier",
        should_iterate,
        {
            "finalize": "finalize",
            "classify": "analyze_gaps"
        }
    )
    

    g.add_edge("analyze_gaps", "smart_prioritizer")
    g.add_edge("smart_prioritizer", "supplement_tests")
    g.add_edge("supplement_tests", "append_tests")  # Direct connection now
    g.add_edge("append_tests", "execute")
    
    # Exit
    g.add_edge("finalize", END)
    
    logger.info("Graph built")
    
    return g.compile()
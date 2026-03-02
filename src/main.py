import logging
import os
import re 
from dotenv import load_dotenv
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree

from schema import AgentState, Settings
from graph import build_graph
from langsmith_utils import (
    is_langsmith_enabled, 
    create_run_summary,
    log_metric
)

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

load_dotenv()


@traceable(
    name="unit_test_generator_pipeline",
    tags=["pipeline", "main", "test-generation"],
    metadata={"version": "3.0", "phase": "1"}
)
def run_pipeline(state: AgentState) -> AgentState:
    """
    Main pipeline execution with LangSmith tracing.
    """
    logger.info("Building and executing the processing graph")
    
    # Build and run the graph
    app = build_graph()
    
    # Log initial state
    log_metric("initial_targets", len(state.targets))
    log_metric("initial_source_files", len(state.source_map))
    
    # Execute pipeline
    final_state = app.invoke(state)
    
    # Log final metrics
    summary = create_run_summary(final_state)
    
    # Attach summary to LangSmith run
    run_tree = get_current_run_tree()
    if run_tree:
        run_tree.extra = run_tree.extra or {}
        run_tree.extra["pipeline_summary"] = summary
    
    # Log individual metrics
    for key, value in summary.items():
        log_metric(key, value)
    
    return final_state


def main():
    logger.info("Starting Unit Test Generator")
    
    # Check LangSmith status
    if is_langsmith_enabled():
        logger.info("LangSmith tracing is ENABLED")
        logger.info(f"  Project: {os.getenv('LANGCHAIN_PROJECT', 'default')}")
    else:
        logger.info("âš ï¸  LangSmith tracing is DISABLED")
        logger.info("  Set LANGCHAIN_TRACING_V2=true in .env to enable")
    
    # Load environment variables
    paths_env = os.getenv("SOURCE_ROOT_PATH", "")
    project_root = os.getenv("PROJECT_ROOT", "")
    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    temperature = float(os.getenv("TEMPERATURE", "0.0"))
    max_output_tokens = int(os.getenv("MAX_OUTPUT_TOKENS", "2000"))
    test_root = os.getenv("TEST_ROOT", "tests")
    auto_install_deps = os.getenv("AUTO_INSTALL_DEPS", "true").lower() == "true"  
    pip_timeout = int(os.getenv("PIP_TIMEOUT", "300"))  
    max_iterations = int(os.getenv("MAX_ITERATIONS", "1"))
    coverage_threshold = float(os.getenv("COVERAGE_THRESHOLD", "80.0"))
    
    enable_pattern_classification = os.getenv("ENABLE_PATTERN_CLASSIFICATION", "true").lower() == "true"
    enable_smart_prioritization = os.getenv("ENABLE_SMART_PRIORITIZATION", "true").lower() == "true"
    max_prioritized_targets = int(os.getenv("MAX_PRIORITIZED_TARGETS", "10"))
    
    logger.debug(f"Configuration: model={model}, temperature={temperature}, "
                f"max_tokens={max_output_tokens}, test_root={test_root}, "
                f"auto_install_deps={auto_install_deps}")
    
    logger.info(f"Intelligence Layer: pattern_classification={enable_pattern_classification}, "
               f"smart_prioritization={enable_smart_prioritization}")

    # Parse source paths
    paths = [p for p in (re.split(r"[,\s;]+", paths_env) if paths_env else []) if p]
    if paths:
        logger.info(f"Source paths to process: {', '.join(paths)}")
    else:
        logger.warning("No source paths specified")

    # Initialize settings and state
    logger.debug("Initializing settings and state")
    settings = Settings(
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        test_root=test_root,
        project_root=project_root,
        auto_install_deps=auto_install_deps,
        pip_timeout=pip_timeout,
        enable_pattern_classification=enable_pattern_classification,
        enable_smart_prioritization=enable_smart_prioritization,
        max_prioritized_targets=max_prioritized_targets,
    )

    state = AgentState(
        paths=paths,
        settings=settings,
        max_iterations=max_iterations,
        coverage_threshold=coverage_threshold,
    )

    # Run pipeline with tracing
    final_state = run_pipeline(state)

    # Final summary
    logger.info("=== Execution Summary ===")
    discovered_files = len(final_state.get('source_map', {}))
    targets_found = len(final_state.get('targets', []))
    planned_cases = len(final_state.get('plan', []))
    generated_files = len(final_state.get('generated_tests', {}))
    
    logger.info(f"Discovered files: {discovered_files}")
    logger.info(f"Targets found: {targets_found}")
    logger.info(f"Planned cases: {planned_cases}")
    logger.info(f"Generated tests files: {generated_files}")
    
    # Dependency summary
    detected_deps = len(final_state.get('detected_dependencies', []))
    installed_deps = len(final_state.get('installed_dependencies', []))
    failed_deps = len(final_state.get('failed_dependencies', []))
    
    if detected_deps > 0:
        logger.info(f"Dependencies detected: {detected_deps}")
        logger.info(f"Dependencies installed: {installed_deps}")
        if failed_deps > 0:
            logger.warning(f"Dependencies failed: {failed_deps}")
            logger.warning(f"Failed packages: {final_state.get('failed_dependencies', [])}")
    
    logger.info(f"Tests written under: {test_root}/")
    
    # Coverage summary
    coverage_report = final_state.get('coverage_report', {})
    if coverage_report and 'summary' in coverage_report:
        summary = coverage_report['summary']
        if 'summary' in summary:
            inner = summary['summary']
            logger.info(f"Coverage: {inner.get('percent', 0):.1f}% "
                       f"({inner.get('covered', 0)}/{inner.get('statements', 0)} statements)")
    
    patterns = final_state.get('coverage_patterns', [])
    prioritized = final_state.get('prioritized_targets', [])
    
    if enable_pattern_classification and patterns:
        logger.info(f"Coverage patterns identified: {len(patterns)}")
    
    if enable_smart_prioritization and prioritized:
        logger.info(f"Targets prioritized: {len(prioritized)}")
        skipped = final_state.get('skipped_targets', [])
        if skipped:
            logger.info(f"Targets skipped (low ROI): {len(skipped)}")
    
    # LangSmith link
    if is_langsmith_enabled():
        logger.info("=" * 60)
        logger.info("View detailed trace in LangSmith:")
        logger.info(f"  https://smith.langchain.com/")
        logger.info("=" * 60)
    
    logger.info("Unit Test Generator completed successfully")


if __name__ == "__main__":
    main()
"""
Finalizer - Generates comprehensive summary report.

This node:
1. Aggregates all metrics from the pipeline
2. Calculates improvement statistics
3. Identifies functions still needing work
4. Generates JSON and markdown reports
5. Logs actionable insights
"""

import json
import logging
import os
from typing import Dict, Any, List
from datetime import datetime
from schema import AgentState

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _calculate_coverage_stats(state: AgentState) -> Dict[str, Any]:
    """Extract coverage statistics from coverage report."""
    coverage_report = state.coverage_report
    
    if not coverage_report:
        return {
            'overall_percent': 0,
            'statements_total': 0,
            'statements_covered': 0,
            'statements_missing': 0,
        }
    
    outer_summary = coverage_report.get('summary', {})
    
    if isinstance(outer_summary, dict):
        # Get the inner summary
        inner_summary = outer_summary.get('summary', {})
        if inner_summary:
            return {
                'overall_percent': inner_summary.get('percent', 0),
                'statements_total': inner_summary.get('statements', 0),
                'statements_covered': inner_summary.get('covered', 0),
                'statements_missing': inner_summary.get('missing', 0),
            }
    
    # Fallback: try direct access
    return {
        'overall_percent': 0,
        'statements_total': 0,
        'statements_covered': 0,
        'statements_missing': 0,
    }


def _calculate_test_stats(state: AgentState) -> Dict[str, Any]:
    """Extract test execution statistics."""
    exec_report = state.execution_report
    
    if not exec_report or 'summary' not in exec_report:
        return {
            'total': 0,
            'passed': 0,
            'failed': 0,
            'errors': 0,
            'skipped': 0,
        }
    
    summary = exec_report.get('summary', {})
    
    total = summary.get('tests', 0)
    failures = summary.get('failures', 0)
    errors = summary.get('errors', 0)
    skipped = summary.get('skipped', 0)
    passed = total - failures - errors - skipped
    
    return {
        'total': total,
        'passed': passed,
        'failed': failures,
        'errors': errors,
        'skipped': skipped,
    }


def _calculate_per_file_stats(state: AgentState) -> List[Dict[str, Any]]:
    """Extract per-file coverage statistics."""
    coverage_report = state.coverage_report
    
    if not coverage_report:
        return []
    
    # FIXED: Handle nested structure
    outer_summary = coverage_report.get('summary', {})
    if not isinstance(outer_summary, dict):
        return []
    
    per_file = outer_summary.get('per_file', {})
    if not per_file:
        return []
    
    file_stats = []
    for file_path, stats in per_file.items():
        file_stats.append({
            'file': os.path.basename(file_path),
            'path': file_path,
            'percent': stats.get('percent', 0),
            'covered': stats.get('covered', 0),
            'statements': stats.get('statements', 0),
            'missing': stats.get('missing', 0),
            'status': 'PASS' if stats.get('percent', 0) >= state.coverage_threshold else 'NEEDS WORK',
        })
    
    # Sort by coverage ascending (worst first)
    file_stats.sort(key=lambda x: x['percent'])
    
    return file_stats


def _identify_failing_functions(state: AgentState) -> List[Dict[str, Any]]:
    """
    Identify functions that still have low coverage after all iterations.
    """
    failing = []
    
    # If we have failing_functions from gap analysis, use that
    if state.failing_functions:
        for func in state.failing_functions:
            if func['coverage'] < state.coverage_threshold:
                failing.append({
                    'name': func['name'],
                    'module': func['module'],
                    'coverage': func['coverage'],
                    'uncovered_lines': len(func['uncovered_lines']),
                })
    
    return failing


def _generate_improvement_summary(state: AgentState) -> Dict[str, Any]:
    """
    Calculate improvement metrics if we have previous coverage.
    """
    # FIXED: Use the corrected extraction function
    current_coverage = _calculate_coverage_stats(state)['overall_percent']
    previous_coverage = state.previous_coverage if state.previous_coverage else None
    
    if previous_coverage is None:
        return {
            'improved': False,
            'previous_coverage': None,
            'current_coverage': current_coverage,
            'improvement': 0,
        }
    
    improvement = current_coverage - previous_coverage
    
    return {
        'improved': improvement > 0,
        'previous_coverage': previous_coverage,
        'current_coverage': current_coverage,
        'improvement': improvement,
    }


def _save_json_report(report: Dict[str, Any], output_path: str) -> None:
    """Save report as JSON file."""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Saved JSON report: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save JSON report: {e}")


def _save_markdown_report(report: Dict[str, Any], output_path: str) -> None:
    """Save report as Markdown file."""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# Test Generation Summary Report\n\n")
            f.write(f"**Generated:** {report['metadata']['timestamp']}\n\n")
            
            # Coverage Summary
            f.write("## Coverage Summary\n\n")
            cov = report['coverage']
            f.write(f"- **Overall Coverage:** {cov['overall_percent']:.1f}%\n")
            f.write(f"- **Statements Covered:** {cov['statements_covered']} / {cov['statements_total']}\n")
            f.write(f"- **Statements Missing:** {cov['statements_missing']}\n\n")
            
            # Test Execution
            f.write("## Test Execution\n\n")
            tests = report['tests']
            f.write(f"- **Total Tests:** {tests['total']}\n")
            f.write(f"- **Passed:** {tests['passed']} ✓\n")
            f.write(f"- **Failed:** {tests['failed']}\n")
            f.write(f"- **Errors:** {tests['errors']}\n")
            f.write(f"- **Skipped:** {tests['skipped']}\n\n")
            
            # Iterations
            f.write("## Iterations\n\n")
            f.write(f"- **Iterations Executed:** {report['iterations']['count']}\n")
            f.write(f"- **Max Iterations:** {report['iterations']['max']}\n")
            
            if report['iterations']['improvement']:
                imp = report['iterations']['improvement']
                f.write(f"- **Previous Coverage:** {imp['previous_coverage']:.1f}%\n")
                f.write(f"- **Improvement:** {imp['improvement']:+.1f}%\n\n")
            
            # Per-File Coverage
            f.write("## Per-File Coverage\n\n")
            f.write("| File | Coverage | Status |\n")
            f.write("|------|----------|--------|\n")
            
            for file in report['per_file_coverage']:
                status_emoji = "✓" if file['status'] == 'PASS' else "⚠️"
                f.write(f"| {file['file']} | {file['percent']:.1f}% | {status_emoji} {file['status']} |\n")
            
            f.write("\n")
            
            # Functions Needing Work
            if report['failing_functions']:
                f.write("## Functions Still Needing Work\n\n")
                f.write("These functions did not reach the coverage threshold:\n\n")
                
                for func in report['failing_functions']:
                    f.write(f"- **{func['module']}.{func['name']}**: {func['coverage']:.1f}% "
                           f"({func['uncovered_lines']} uncovered lines)\n")
                
                f.write("\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            
            if cov['overall_percent'] >= report['metadata']['coverage_threshold']:
                f.write("✓ **Coverage goal achieved!** All targets met the threshold.\n\n")
            else:
                f.write("⚠️ **Coverage goal not met.** Consider:\n\n")
                f.write("- Manually reviewing functions with low coverage\n")
                f.write("- Adding integration tests for complex flows\n")
                f.write("- Using `# pragma: no cover` for unreachable code\n")
                f.write("- Increasing max_iterations in configuration\n\n")
        
        logger.info(f"Saved Markdown report: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save Markdown report: {e}")


def finalize_node(state: AgentState) -> AgentState:
    """
    Generate comprehensive summary report.
    
    Aggregates all metrics and creates:
    1. JSON report (machine-readable)
    2. Markdown report (human-readable)
    3. Console summary with actionable insights
    """
    logger.info("=" * 70)
    logger.info("GENERATING FINAL REPORT")
    logger.info("=" * 70)
    
    # Gather all statistics
    coverage_stats = _calculate_coverage_stats(state)
    test_stats = _calculate_test_stats(state)
    per_file_stats = _calculate_per_file_stats(state)
    failing_functions = _identify_failing_functions(state)
    improvement_summary = _generate_improvement_summary(state)
    
    # Build comprehensive report
    report = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'project_root': state.settings.project_root,
            'test_root': state.settings.test_root,
            'coverage_threshold': state.coverage_threshold,
        },
        'coverage': coverage_stats,
        'tests': test_stats,
        'iterations': {
            'count': state.iteration_count,
            'max': state.max_iterations,
            'improvement': improvement_summary,
        },
        'per_file_coverage': per_file_stats,
        'failing_functions': failing_functions,
        'dependencies': {
            'detected': len(state.detected_dependencies),
            'installed': len(state.installed_dependencies),
            'failed': len(state.failed_dependencies),
        },
        'files_processed': {
            'source_files': len(state.source_map),
            'targets_found': len(state.targets),
            'test_files_generated': len(state.generated_tests),
        }
    }
    
    # Save reports
    reports_dir = os.path.join(os.getcwd(), "reports")
    os.makedirs(reports_dir, exist_ok=True)
    
    json_path = os.path.join(reports_dir, "summary.json")
    md_path = os.path.join(reports_dir, "summary.md")
    
    _save_json_report(report, json_path)
    _save_markdown_report(report, md_path)
    
    # Update state
    state.final_report = report
    
    # Console summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 70)
    
    # Coverage
    cov = coverage_stats
    threshold = state.coverage_threshold
    goal_met = cov['overall_percent'] >= threshold
    
    logger.info(f"Coverage: {cov['overall_percent']:.1f}% "
               f"({cov['statements_covered']}/{cov['statements_total']} statements)")
    
    if goal_met:
        logger.info(f"✓ Coverage goal MET (≥{threshold}%)")
    else:
        logger.warning(f"✗ Coverage goal NOT MET (<{threshold}%)")
    
    # Tests
    tests = test_stats
    logger.info(f"Tests: {tests['passed']}/{tests['total']} passed")
    
    if tests['failed'] > 0:
        logger.warning(f"  {tests['failed']} failed")
    if tests['errors'] > 0:
        logger.warning(f"  {tests['errors']} errors")
    
    # Iterations
    logger.info(f"Iterations: {state.iteration_count}/{state.max_iterations}")
    
    if improvement_summary['improved']:
        imp = improvement_summary['improvement']
        logger.info(f"  Coverage improved by {imp:+.1f}%")
    elif state.iteration_count > 0:
        logger.warning("  No coverage improvement in iteration")
    
    # Files needing work
    files_below = [f for f in per_file_stats if f['status'] == 'NEEDS WORK']
    if files_below:
        logger.warning(f"Files below threshold: {len(files_below)}")
        for f in files_below[:5]:  # Show top 5
            logger.warning(f"  • {f['file']}: {f['percent']:.1f}%")
        if len(files_below) > 5:
            logger.warning(f"  ... and {len(files_below) - 5} more")
    
    # Functions needing work
    if failing_functions:
        logger.warning(f"Functions needing work: {len(failing_functions)}")
        for func in failing_functions[:5]:  # Show top 5
            logger.warning(f"  • {func['module']}.{func['name']}: {func['coverage']:.1f}%")
        if len(failing_functions) > 5:
            logger.warning(f"  ... and {len(failing_functions) - 5} more")
    
    logger.info("=" * 70)
    logger.info("Reports saved:")
    logger.info(f"  • JSON: {json_path}")
    logger.info(f"  • Markdown: {md_path}")
    logger.info("=" * 70)
    
    return state
"""
Gap Analyzer - UNIFIED handling for functions and methods.
"""

import ast
import json
import logging
import os
from typing import Dict, List, Any, Optional

from schema import AgentState

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _read_coverage_json(coverage_json_path: str) -> Dict[str, Any]:
    """Read and parse coverage.json file."""
    if not os.path.exists(coverage_json_path):
        logger.error(f"Coverage JSON not found: {coverage_json_path}")
        return {}
    
    try:
        with open(coverage_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            logger.debug(f"Loaded coverage data with keys: {list(data.keys())}")
            return data
    except Exception as e:
        logger.error(f"Failed to read coverage JSON: {e}")
        return {}


def _get_files_below_threshold(coverage_data: Dict[str, Any], threshold: float) -> Dict[str, Any]:
    """Extract files with coverage below threshold."""
    files_below = {}
    
    files = coverage_data.get('files', {})
    for file_path, file_data in files.items():
        summary = file_data.get('summary', {})
        percent = summary.get('percent_covered', 100)
        
        if percent < threshold:
            files_below[file_path] = {
                'percent': percent,
                'missing_lines': file_data.get('missing_lines', []),
                'executed_lines': file_data.get('executed_lines', []),
            }
            logger.debug(f"Found failing file: {file_path} ({percent:.1f}%)")
    
    return files_below


def _map_lines_to_targets(source_code: str, missing_lines: List[int]) -> Dict[str, List[int]]:
    """
    UNIFIED coverage mapper for functions and methods.
    
    Returns: {target_name: [uncovered_lines]}
    
    Where target_name can be:
    - "function_name" for standalone functions
    - "ClassName.method_name" for methods
    """
    try:
        tree = ast.parse(source_code)
    except SyntaxError as e:
        logger.error(f"Syntax error parsing source: {e}")
        return {}
    
    target_lines = {}  # {name: (start_line, end_line)}
    
    # Single traversal handles everything
    for node in tree.body:
        # Top-level functions
        if isinstance(node, ast.FunctionDef):
            if not node.name.startswith('_'):
                start_line = node.lineno
                end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line
                target_lines[node.name] = (start_line, end_line)
        
        # Class methods
        elif isinstance(node, ast.ClassDef):
            class_name = node.name
            if not class_name.startswith('_'):
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        if not item.name.startswith('_'):
                            # Use dot notation
                            target_name = f"{class_name}.{item.name}"
                            start_line = item.lineno
                            end_line = item.end_lineno if hasattr(item, 'end_lineno') else start_line
                            target_lines[target_name] = (start_line, end_line)
    
    # Map missing lines to targets
    target_missing = {}
    
    for target_name, (start, end) in target_lines.items():
        uncovered = [line for line in missing_lines if start <= line <= end]
        if uncovered:
            target_missing[target_name] = sorted(uncovered)
            logger.debug(f"Target '{target_name}' has {len(uncovered)} uncovered lines")
    
    return target_missing


def _extract_source_for_target(source_code: str, target_name: str) -> Optional[str]:
    """
    UNIFIED source extractor for functions and methods.
    
    Args:
        target_name: Either "function_name" or "ClassName.method_name"
    """
    try:
        tree = ast.parse(source_code)
        
        # Detect type by presence of dot
        if '.' in target_name:
            # Method
            class_name, method_name = target_name.split('.', 1)
            
            for node in tree.body:
                if isinstance(node, ast.ClassDef) and node.name == class_name:
                    # Return full class + highlighted method
                    method_node = None
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef) and item.name == method_name:
                            method_node = item
                            break
                    
                    class_source = ast.unparse(node)
                    method_source = ast.unparse(method_node) if method_node else "# Method not found"
                    
                    return f"""# Full class (for instantiation):
{class_source}

# Target method:
{method_source}"""
            
            logger.warning(f"Method '{target_name}' not found in source")
            return None
        
        else:
            # Function
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and node.name == target_name:
                    return ast.unparse(node)
            
            logger.warning(f"Function '{target_name}' not found in source")
            return None
        
    except Exception as e:
        logger.error(f"Failed to extract source: {e}")
        return None


def _extract_existing_tests(test_file_path: str, target_name: str) -> str:
    """
    Extract existing tests for target (function or method).
    """
    if not os.path.exists(test_file_path):
        logger.warning(f"Test file not found: {test_file_path}")
        return "# No existing tests found"
    
    try:
        with open(test_file_path, 'r', encoding='utf-8') as f:
            test_code = f.read()
        
        tree = ast.parse(test_code)
        existing_tests = []
        
        # Build search pattern
        if '.' in target_name:
            # Method: ClassName.method -> look for test_ClassName_method*
            search_pattern = target_name.replace('.', '_')
        else:
            # Function: function_name -> look for test_function_name*
            search_pattern = target_name
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if node.name.startswith(f"test_{search_pattern}"):
                    test_source = ast.unparse(node)
                    existing_tests.append(test_source)
        
        if existing_tests:
            result = "\n\n".join(existing_tests)
            logger.debug(f"Found {len(existing_tests)} existing tests for {target_name}")
            return result
        else:
            return f"# No existing tests found for {target_name}"
            
    except Exception as e:
        logger.error(f"Failed to extract existing tests: {e}")
        return "# Error extracting existing tests"


def _convert_file_path_to_module(file_path: str, project_root: str) -> str:
    """Convert absolute file path to Python module path."""
    abs_path = os.path.abspath(file_path)
    abs_root = os.path.abspath(project_root)
    
    try:
        rel_path = os.path.relpath(abs_path, abs_root)
        module = rel_path.replace(os.sep, '.')
        if module.endswith('.py'):
            module = module[:-3]
        return module
    except ValueError:
        basename = os.path.basename(file_path)
        if basename.endswith('.py'):
            basename = basename[:-3]
        return basename


def analyze_gaps_node(state: AgentState) -> AgentState:
    """
    Analyze coverage gaps - works for both functions and methods.
    """
    logger.info("=" * 60)
    logger.info("Starting Gap Analysis")
    logger.info("=" * 60)
    
    coverage_json_path = state.coverage_json_path
    if not coverage_json_path:
        logger.error("No coverage JSON path in state")
        state.failing_functions = []
        return state
    
    coverage_data = _read_coverage_json(coverage_json_path)
    if not coverage_data:
        logger.error("Failed to load coverage data")
        state.failing_functions = []
        return state
    
    threshold = state.coverage_threshold
    files_below = _get_files_below_threshold(coverage_data, threshold)
    
    logger.info(f"Found {len(files_below)} files below {threshold}% coverage")
    
    if not files_below:
        logger.info("No files need improvement")
        state.failing_functions = []
        return state
    
    failing_functions = []
    
    for file_path, coverage_info in files_below.items():
        logger.info(f"Analyzing: {file_path} ({coverage_info['percent']:.1f}%)")
        
        module_path = _convert_file_path_to_module(file_path, state.settings.project_root)
        
        source_code = state.source_map.get(module_path)
        if not source_code:
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        source_code = f.read()
                except Exception as e:
                    logger.error(f"Failed to read {file_path}: {e}")
                    continue
            else:
                logger.warning(f"Source code not found for {module_path}")
                continue
        
        missing_lines = coverage_info['missing_lines']
        target_missing = _map_lines_to_targets(source_code, missing_lines)
        
        if not target_missing:
            logger.warning(f"Could not map uncovered lines to targets in {module_path}")
            continue
        
        for target_name, uncovered_lines in target_missing.items():
            target_source = _extract_source_for_target(source_code, target_name)
            if not target_source:
                continue
        
        # Calculate coverage
        target_total_lines = len(target_source.split('\n'))
        target_covered_lines = target_total_lines - len(uncovered_lines)
        target_coverage = (target_covered_lines / target_total_lines * 100) if target_total_lines > 0 else 0
        
        existing_tests = _extract_existing_tests(state.test_file, target_name)
        
        failing_func = {
            'name': target_name,
            'module': module_path,
            'file_path': file_path,
            'uncovered_lines': uncovered_lines,
            'coverage': target_coverage,
            'source': target_source,
            'existing_tests': existing_tests,
        }
        
        failing_functions.append(failing_func)
        
        logger.info(f"  ⚠️  {target_name}: {target_coverage:.1f}% coverage, "
                   f"{len(uncovered_lines)} uncovered lines")

        state.failing_functions = failing_functions
        state.iteration_count += 1

        # Extract coverage percentage safely
        coverage_report = state.coverage_report
        if coverage_report:
            outer_summary = coverage_report.get('summary', {})
            if isinstance(outer_summary, dict):
                inner_summary = outer_summary.get('summary', {})
                if inner_summary:
                    state.previous_coverage = inner_summary.get('percent', 0)
                else:
                    state.previous_coverage = outer_summary.get('percent', 0)

        logger.info("=" * 60)
        logger.info(f"Gap Analysis Complete")
        logger.info(f"  Functions needing improvement: {len(failing_functions)}")
        logger.info(f"  Starting iteration: {state.iteration_count}")
        logger.info("=" * 60)

    return state
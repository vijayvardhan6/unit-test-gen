"""
Pattern Classifier - Analyzes coverage gaps and classifies patterns.

1. Analyzes uncovered lines in source code
2. Uses AST to understand code structure
3. Classifies patterns: branches, exceptions, edge cases, loops, etc.
4. Assigns confidence scores and priorities
5. Provides test generation hints
"""

import ast
import logging
import os
from typing import Dict, List, Set, Optional, Tuple
from schema import AgentState, CoveragePattern

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class PatternDetector(ast.NodeVisitor):
    """AST visitor to detect coverage patterns."""
    
    def __init__(self, uncovered_lines: Set[int]):
        self.uncovered_lines = uncovered_lines
        self.patterns: List[Dict] = []
        self.current_function = None
        
    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Track current function context."""
        old_function = self.current_function
        self.current_function = node.name
        self.generic_visit(node)
        self.current_function = old_function
    
    def visit_If(self, node: ast.If):
        """Detect uncovered branch patterns."""
        # Check if condition or branches are uncovered
        if_line = node.lineno
        
        # Check if-body
        if_body_lines = set()
        for stmt in node.body:
            if hasattr(stmt, 'lineno'):
                if_body_lines.add(stmt.lineno)
        
        # Check else-body
        else_body_lines = set()
        if node.orelse:
            for stmt in node.orelse:
                if hasattr(stmt, 'lineno'):
                    else_body_lines.add(stmt.lineno)
        
        # Pattern: Uncovered if-branch
        if if_body_lines & self.uncovered_lines:
            self.patterns.append({
                'type': 'branch',
                'subtype': 'if_branch',
                'line': if_line,
                'uncovered_lines': sorted(if_body_lines & self.uncovered_lines),
                'context': f"if statement at line {if_line}",
                'suggestion': "Test condition where if-branch executes",
                'complexity': len(if_body_lines),
            })
        
        # Pattern: Uncovered else-branch
        if else_body_lines & self.uncovered_lines:
            self.patterns.append({
                'type': 'branch',
                'subtype': 'else_branch',
                'line': if_line,
                'uncovered_lines': sorted(else_body_lines & self.uncovered_lines),
                'context': f"else clause at line {if_line}",
                'suggestion': "Test condition where else-branch executes",
                'complexity': len(else_body_lines),
            })
        
        self.generic_visit(node)
    
    def visit_Try(self, node: ast.Try):
        """Detect uncovered exception patterns."""
        try_line = node.lineno
        
        # Check except handlers
        for handler in node.handlers:
            handler_lines = set()
            for stmt in handler.body:
                if hasattr(stmt, 'lineno'):
                    handler_lines.add(stmt.lineno)
            
            if handler_lines & self.uncovered_lines:
                exception_type = ast.unparse(handler.type) if handler.type else "Exception"
                self.patterns.append({
                    'type': 'exception',
                    'subtype': 'except_handler',
                    'line': handler.lineno,
                    'uncovered_lines': sorted(handler_lines & self.uncovered_lines),
                    'context': f"except {exception_type} handler",
                    'suggestion': f"Mock/trigger {exception_type} to test handler",
                    'complexity': len(handler_lines),
                })
        
        # Check else clause
        if node.orelse:
            else_lines = set()
            for stmt in node.orelse:
                if hasattr(stmt, 'lineno'):
                    else_lines.add(stmt.lineno)
            
            if else_lines & self.uncovered_lines:
                self.patterns.append({
                    'type': 'exception',
                    'subtype': 'try_else',
                    'line': try_line,
                    'uncovered_lines': sorted(else_lines & self.uncovered_lines),
                    'context': "try-else clause (no exception path)",
                    'suggestion': "Test successful execution without exceptions",
                    'complexity': len(else_lines),
                })
        
        # Check finally clause
        if node.finalbody:
            finally_lines = set()
            for stmt in node.finalbody:
                if hasattr(stmt, 'lineno'):
                    finally_lines.add(stmt.lineno)
            
            if finally_lines & self.uncovered_lines:
                self.patterns.append({
                    'type': 'exception',
                    'subtype': 'finally',
                    'line': try_line,
                    'uncovered_lines': sorted(finally_lines & self.uncovered_lines),
                    'context': "finally clause",
                    'suggestion': "Ensure finally block executes",
                    'complexity': len(finally_lines),
                })
        
        self.generic_visit(node)
    
    def visit_For(self, node: ast.For):
        """Detect uncovered loop patterns."""
        loop_line = node.lineno
        
        # Check loop body
        body_lines = set()
        for stmt in node.body:
            if hasattr(stmt, 'lineno'):
                body_lines.add(stmt.lineno)
        
        if body_lines & self.uncovered_lines:
            self.patterns.append({
                'type': 'loop',
                'subtype': 'for_loop',
                'line': loop_line,
                'uncovered_lines': sorted(body_lines & self.uncovered_lines),
                'context': f"for loop at line {loop_line}",
                'suggestion': "Test with non-empty iterable",
                'complexity': len(body_lines),
            })
        
        # Check else clause (loop not broken)
        if node.orelse:
            else_lines = set()
            for stmt in node.orelse:
                if hasattr(stmt, 'lineno'):
                    else_lines.add(stmt.lineno)
            
            if else_lines & self.uncovered_lines:
                self.patterns.append({
                    'type': 'loop',
                    'subtype': 'for_else',
                    'line': loop_line,
                    'uncovered_lines': sorted(else_lines & self.uncovered_lines),
                    'context': "for-else clause",
                    'suggestion': "Test loop that completes without break",
                    'complexity': len(else_lines),
                })
        
        self.generic_visit(node)
    
    def visit_While(self, node: ast.While):
        """Detect uncovered while loop patterns."""
        loop_line = node.lineno
        
        body_lines = set()
        for stmt in node.body:
            if hasattr(stmt, 'lineno'):
                body_lines.add(stmt.lineno)
        
        if body_lines & self.uncovered_lines:
            self.patterns.append({
                'type': 'loop',
                'subtype': 'while_loop',
                'line': loop_line,
                'uncovered_lines': sorted(body_lines & self.uncovered_lines),
                'context': f"while loop at line {loop_line}",
                'suggestion': "Test condition that enters loop",
                'complexity': len(body_lines),
            })
        
        self.generic_visit(node)
    
    def visit_Raise(self, node: ast.Raise):
        """Detect uncovered raise statements."""
        if node.lineno in self.uncovered_lines:
            exception_type = ast.unparse(node.exc) if node.exc else "Exception"
            self.patterns.append({
                'type': 'exception',
                'subtype': 'raise_statement',
                'line': node.lineno,
                'uncovered_lines': [node.lineno],
                'context': f"raises {exception_type}",
                'suggestion': f"Test input that triggers {exception_type}",
                'complexity': 1,
            })
        
        self.generic_visit(node)
    
    def visit_Return(self, node: ast.Return):
        """Detect uncovered return paths."""
        if node.lineno in self.uncovered_lines:
            self.patterns.append({
                'type': 'branch',
                'subtype': 'return_statement',
                'line': node.lineno,
                'uncovered_lines': [node.lineno],
                'context': "return statement",
                'suggestion': "Test code path that reaches this return",
                'complexity': 1,
            })
        
        self.generic_visit(node)


def _detect_edge_case_patterns(
    source_code: str, 
    uncovered_lines: Set[int],
    function_name: str
) -> List[Dict]:
    """Detect potential edge case patterns by analyzing code."""
    patterns = []
    lines = source_code.split('\n')
    
    for line_no in uncovered_lines:
        if line_no > len(lines):
            continue
        
        line = lines[line_no - 1].strip()
        
        # Check for common edge case indicators
        if any(keyword in line.lower() for keyword in ['none', 'null', 'empty']):
            patterns.append({
                'type': 'edge_case',
                'subtype': 'none_check',
                'line': line_no,
                'uncovered_lines': [line_no],
                'context': f"None/null handling at line {line_no}",
                'suggestion': "Test with None/null values",
                'complexity': 1,
            })
        
        if any(keyword in line for keyword in ['==', '!=', '>', '<', '>=', '<=']):
            patterns.append({
                'type': 'edge_case',
                'subtype': 'boundary_condition',
                'line': line_no,
                'uncovered_lines': [line_no],
                'context': f"Boundary check at line {line_no}",
                'suggestion': "Test boundary values (0, -1, max)",
                'complexity': 1,
            })
        
        if 'len(' in line or '.count(' in line or '.size' in line:
            patterns.append({
                'type': 'edge_case',
                'subtype': 'size_check',
                'line': line_no,
                'uncovered_lines': [line_no],
                'context': f"Size/length check at line {line_no}",
                'suggestion': "Test with empty collections",
                'complexity': 1,
            })
    
    return patterns


def _detect_state_mutation_patterns(
    source_code: str,
    uncovered_lines: Set[int],
    is_class_method: bool
) -> List[Dict]:
    """Detect state mutation patterns in class methods."""
    if not is_class_method:
        return []
    
    patterns = []
    lines = source_code.split('\n')
    
    for line_no in uncovered_lines:
        if line_no > len(lines):
            continue
        
        line = lines[line_no - 1].strip()
        
        # Check for self attribute assignments
        if line.startswith('self.') and '=' in line:
            patterns.append({
                'type': 'state_mutation',
                'subtype': 'attribute_assignment',
                'line': line_no,
                'uncovered_lines': [line_no],
                'context': f"State mutation at line {line_no}",
                'suggestion': "Test method and verify state change",
                'complexity': 2,
            })
    
    return patterns


def _calculate_pattern_confidence(pattern: Dict) -> float:
    """Calculate confidence score for a pattern (0-1)."""
    # Base confidence by pattern type
    confidence_base = {
        'branch': 0.9,
        'exception': 0.85,
        'edge_case': 0.7,
        'loop': 0.8,
        'state_mutation': 0.75,
    }
    
    base = confidence_base.get(pattern['type'], 0.5)
    
    # Adjust by complexity (simpler = more confident)
    complexity_penalty = min(pattern['complexity'] * 0.05, 0.2)
    
    # Adjust by number of uncovered lines (fewer = more confident)
    lines_penalty = min(len(pattern['uncovered_lines']) * 0.02, 0.15)
    
    confidence = base - complexity_penalty - lines_penalty
    
    return max(0.3, min(1.0, confidence))


def _calculate_pattern_priority(pattern: Dict, current_coverage: float) -> int:
    """Calculate priority (1-5, 5=highest)."""
    # Factors:
    # 1. Pattern type importance
    # 2. Number of uncovered lines
    # 3. Current coverage (lower = higher priority)
    
    type_priority = {
        'exception': 5,  # Exception handling is critical
        'branch': 4,     # Branches are important
        'loop': 3,       # Loops are moderate
        'edge_case': 3,  # Edge cases are moderate
        'state_mutation': 2,  # State changes are lower
    }
    
    base_priority = type_priority.get(pattern['type'], 2)
    
    # Boost if many lines uncovered
    if len(pattern['uncovered_lines']) > 5:
        base_priority = min(5, base_priority + 1)
    
    # Boost if coverage is very low
    if current_coverage < 50:
        base_priority = min(5, base_priority + 1)
    
    return base_priority


def _extract_code_context(source_code: str, line_number: int, context_lines: int = 3) -> str:
    """Extract code snippet around a line number."""
    lines = source_code.split('\n')
    start = max(0, line_number - context_lines - 1)
    end = min(len(lines), line_number + context_lines)
    
    context = []
    for i in range(start, end):
        marker = " → " if i == line_number - 1 else "   "
        context.append(f"{i+1:3d}{marker}{lines[i]}")
    
    return '\n'.join(context)


def _classify_patterns_for_target(
    target_name: str,
    module_path: str,
    source_code: str,
    uncovered_lines: List[int],
    current_coverage: float,
    is_method: bool = False
) -> List[CoveragePattern]:
    """Classify coverage patterns for a single target."""
    if not uncovered_lines:
        return []
    
    uncovered_set = set(uncovered_lines)
    patterns = []
    
    # Parse source code
    try:
        tree = ast.parse(source_code)
    except SyntaxError as e:
        logger.error(f"Syntax error parsing {target_name}: {e}")
        return []
    
    # Run AST-based pattern detection
    detector = PatternDetector(uncovered_set)
    detector.visit(tree)
    
    # Add edge case patterns
    edge_patterns = _detect_edge_case_patterns(source_code, uncovered_set, target_name)
    detector.patterns.extend(edge_patterns)
    
    # Add state mutation patterns (for methods)
    if is_method:
        state_patterns = _detect_state_mutation_patterns(source_code, uncovered_set, True)
        detector.patterns.extend(state_patterns)
    
    # Convert to CoveragePattern objects
    for pattern_dict in detector.patterns:
        confidence = _calculate_pattern_confidence(pattern_dict)
        priority = _calculate_pattern_priority(pattern_dict, current_coverage)
        
        # Extract code context
        main_line = pattern_dict['line']
        context = _extract_code_context(source_code, main_line)
        
        pattern = CoveragePattern(
            pattern_type=pattern_dict['type'],
            target_name=target_name,
            module_path=module_path,
            uncovered_lines=pattern_dict['uncovered_lines'],
            confidence=confidence,
            priority=priority,
            suggested_test_approach=pattern_dict['suggestion'],
            complexity_score=float(pattern_dict['complexity']),
            context=context,
        )
        
        patterns.append(pattern)
    
    return patterns


def pattern_classifier_node(state: AgentState) -> AgentState:
    """
    Classify coverage gap patterns using AST analysis.
    """
    if not state.settings.enable_pattern_classification:
        logger.info("Pattern classification disabled in settings")
        return state
    
    logger.info("=" * 70)
    logger.info("PATTERN CLASSIFICATION (Intelligence Layer)")
    logger.info("=" * 70)
    
    coverage_report = state.coverage_report
    
    if not coverage_report or 'summary' not in coverage_report:
        logger.warning("No coverage report available for pattern classification")
        return state
    
    # Extract current coverage
    outer_summary = coverage_report.get('summary', {})
    if isinstance(outer_summary, dict):
        inner_summary = outer_summary.get('summary', {})
        current_coverage = inner_summary.get('percent', 0) if inner_summary else outer_summary.get('percent', 0)
    else:
        current_coverage = 0
    
    logger.info(f"Current coverage: {current_coverage:.1f}%")
    
    # Get files with uncovered lines
    per_file = outer_summary.get('per_file', {})
    
    if not per_file:
        logger.warning("No per-file coverage data available")
        return state
    
    all_patterns = []
    patterns_by_target = {}
    
    # Analyze each file with coverage gaps
    for file_path, file_stats in per_file.items():
        if file_stats.get('percent', 100) >= state.coverage_threshold:
            continue  # Skip files that meet threshold
        
        logger.info(f"Analyzing patterns in: {file_path} ({file_stats['percent']:.1f}%)")
        
        # Get module path
        module_path = file_path.replace('/', '.').replace('\\', '.')
        if module_path.endswith('.py'):
            module_path = module_path[:-3]
        
        # Find module in source_map
        source_code = None
        for mod_key, code in state.source_map.items():
            if mod_key in module_path or module_path in mod_key:
                source_code = code
                module_path = mod_key
                break
        
        if not source_code:
            logger.warning(f"Source code not found for {file_path}")
            continue
        
        # Get uncovered lines from coverage data
        # Note: This requires reading coverage.json for missing_lines
        try:
            import json
            if state.coverage_json_path and os.path.exists(state.coverage_json_path):
                with open(state.coverage_json_path, 'r') as f:
                    cov_data = json.load(f)
                    file_data = cov_data.get('files', {}).get(file_path, {})
                    uncovered_lines = file_data.get('missing_lines', [])
            else:
                logger.warning(f"Cannot read coverage JSON for {file_path}")
                continue
        except Exception as e:
            logger.error(f"Failed to read coverage data: {e}")
            continue
        
        if not uncovered_lines:
            continue
        
        # Classify patterns for each function/method in this file
        # We'll analyze the entire file and let the detector find patterns
        file_patterns = _classify_patterns_for_target(
            target_name=f"{module_path}",
            module_path=module_path,
            source_code=source_code,
            uncovered_lines=uncovered_lines,
            current_coverage=file_stats['percent'],
            is_method=False  # Will detect methods automatically
        )
        
        logger.info(f"  Found {len(file_patterns)} patterns")
        
        # Log pattern summary
        pattern_counts = {}
        for p in file_patterns:
            pattern_counts[p.pattern_type] = pattern_counts.get(p.pattern_type, 0) + 1
        
        for ptype, count in sorted(pattern_counts.items()):
            logger.info(f"    {ptype}: {count}")
        
        all_patterns.extend(file_patterns)
        patterns_by_target[module_path] = file_patterns
    
    # Sort patterns by priority
    all_patterns.sort(key=lambda p: (p.priority, p.confidence), reverse=True)
    
    # Update state
    state.coverage_patterns = all_patterns
    state.patterns_by_target = patterns_by_target
    
    # Summary
    logger.info("=" * 70)
    logger.info(f"Pattern Classification Complete")
    logger.info(f"  Total patterns identified: {len(all_patterns)}")
    
    pattern_type_counts = {}
    for p in all_patterns:
        pattern_type_counts[p.pattern_type] = pattern_type_counts.get(p.pattern_type, 0) + 1
    
    logger.info(f"  Pattern distribution:")
    for ptype, count in sorted(pattern_type_counts.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"    {ptype}: {count}")
    
    high_priority = [p for p in all_patterns if p.priority >= 4]
    logger.info(f"  High priority patterns (4-5): {len(high_priority)}")
    
    high_confidence = [p for p in all_patterns if p.confidence >= 0.8]
    logger.info(f"  High confidence patterns (≥0.8): {len(high_confidence)}")
    
    logger.info("=" * 70)
    
    return state
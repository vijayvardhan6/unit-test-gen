"""
Smart Prioritizer - Selects which targets to focus on for test generation.

1. Takes failing_functions and coverage_patterns
2. Calculates priority scores based on multiple factors
3. Selects top N targets (configurable)
4. Skips low-ROI targets to save LLM calls
5. Provides reasoning for prioritization decisions
"""

import logging
from typing import List, Dict, Any
from schema import AgentState, PrioritizedTarget, CoveragePattern

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _calculate_coverage_impact(
    current_coverage: float,
    uncovered_lines: int,
    total_lines: int
) -> float:
    """
    Calculate potential coverage improvement.
    
    Returns: Estimated % improvement if all lines covered
    """
    if total_lines == 0:
        return 0.0
    
    potential_gain = (uncovered_lines / total_lines) * 100
    
    # Weight by current coverage (lower coverage = higher impact)
    if current_coverage < 50:
        potential_gain *= 1.5
    elif current_coverage < 70:
        potential_gain *= 1.2
    
    return min(potential_gain, 100.0)


def _calculate_pattern_score(patterns: List[CoveragePattern]) -> float:
    """
    Calculate aggregate score from patterns.
    
    Higher score = more important patterns with high confidence.
    """
    if not patterns:
        return 0.0
    
    # Weighted sum: priority * confidence
    total_score = sum(p.priority * p.confidence for p in patterns)
    
    # Normalize by number of patterns (avoid bias toward many small patterns)
    avg_score = total_score / len(patterns)
    
    return avg_score


def _calculate_complexity_factor(patterns: List[CoveragePattern]) -> float:
    """
    Calculate complexity factor (higher = prioritize).
    
    Complex functions need more attention.
    """
    if not patterns:
        return 1.0
    
    # Average complexity across patterns
    avg_complexity = sum(p.complexity_score for p in patterns) / len(patterns)
    
    # Scale: 1.0 (simple) to 2.0 (very complex)
    complexity_factor = min(1.0 + (avg_complexity / 10), 2.0)
    
    return complexity_factor


def _calculate_previous_success_rate(
    target_name: str,
    state: AgentState
) -> float:
    """
    Calculate success rate from previous iterations.
    
    If this target improved in past iterations, prioritize it.
    Returns: 0.5 (no data) to 1.5 (good history)
    """
    # This would require tracking per-target improvements
    # For Phase 1, return neutral value
    # TODO: Implement in Phase 2 with learning
    return 1.0


def _is_on_critical_path(target_name: str, state: AgentState) -> bool:
    """
    Determine if target is on critical path.
    
    Critical path = called by many other functions or is public API.
    """
    # Simple heuristic: public functions (no underscore) are critical
    symbol = target_name.split(':')[-1]
    
    if symbol.startswith('_'):
        return False
    
    # Check if it's a main entry point
    if symbol in ['main', 'run', 'execute', 'process', 'handle']:
        return True
    
    # TODO: Implement dependency graph analysis in Phase 2
    return True  # Default: assume critical


def _calculate_priority_score(
    failing_func: Dict[str, Any],
    patterns: List[CoveragePattern],
    state: AgentState
) -> float:
    """
    Calculate final priority score for a target.
    
    Factors:
    1. Coverage impact (30%)
    2. Pattern importance (30%)
    3. Complexity (20%)
    4. Previous success (10%)
    5. Critical path (10%)
    """
    target_name = failing_func['name']
    current_coverage = failing_func['coverage']
    uncovered_lines = len(failing_func['uncovered_lines'])
    
    # Estimate total lines (rough approximation)
    total_lines = int(uncovered_lines / (1 - current_coverage / 100)) if current_coverage < 100 else uncovered_lines
    
    # Factor 1: Coverage impact (0-10)
    impact = _calculate_coverage_impact(current_coverage, uncovered_lines, total_lines)
    impact_score = min(impact, 10.0)
    
    # Factor 2: Pattern importance (0-10)
    pattern_score = _calculate_pattern_score(patterns)
    pattern_score = min(pattern_score, 10.0)
    
    # Factor 3: Complexity (0-10)
    complexity_factor = _calculate_complexity_factor(patterns)
    complexity_score = complexity_factor * 5  # Scale to 0-10
    
    # Factor 4: Previous success (0-10)
    success_rate = _calculate_previous_success_rate(target_name, state)
    success_score = success_rate * 5  # Scale to 0-10
    
    # Factor 5: Critical path (0-10)
    is_critical = _is_on_critical_path(target_name, state)
    critical_score = 10.0 if is_critical else 3.0
    
    # Weighted sum
    weights = {
        'impact': 0.30,
        'pattern': 0.30,
        'complexity': 0.20,
        'success': 0.10,
        'critical': 0.10,
    }
    
    final_score = (
        impact_score * weights['impact'] +
        pattern_score * weights['pattern'] +
        complexity_score * weights['complexity'] +
        success_score * weights['success'] +
        critical_score * weights['critical']
    )
    
    logger.debug(f"{target_name}: impact={impact_score:.1f}, pattern={pattern_score:.1f}, "
                f"complexity={complexity_score:.1f}, success={success_score:.1f}, "
                f"critical={critical_score:.1f} → TOTAL={final_score:.1f}")
    
    return final_score


def _generate_prioritization_reason(
    failing_func: Dict[str, Any],
    patterns: List[CoveragePattern],
    priority_score: float
) -> str:
    """Generate human-readable explanation for prioritization."""
    reasons = []
    
    # Coverage
    current_cov = failing_func['coverage']
    if current_cov < 50:
        reasons.append(f"very low coverage ({current_cov:.0f}%)")
    elif current_cov < 70:
        reasons.append(f"below target coverage ({current_cov:.0f}%)")
    
    # Patterns
    if patterns:
        high_priority = [p for p in patterns if p.priority >= 4]
        if high_priority:
            reasons.append(f"{len(high_priority)} high-priority patterns")
        
        pattern_types = set(p.pattern_type for p in patterns)
        if 'exception' in pattern_types:
            reasons.append("uncovered exception handling")
        if 'branch' in pattern_types:
            reasons.append("untested branches")
    
    # Complexity
    if patterns:
        avg_complexity = sum(p.complexity_score for p in patterns) / len(patterns)
        if avg_complexity > 5:
            reasons.append("high complexity")
    
    # Lines
    uncovered = len(failing_func['uncovered_lines'])
    if uncovered > 10:
        reasons.append(f"{uncovered} uncovered lines")
    
    if not reasons:
        reasons.append("general coverage improvement needed")
    
    return ", ".join(reasons)


def smart_prioritizer_node(state: AgentState) -> AgentState:
    """
    Intelligently prioritize which targets to focus on for supplemental tests.
    """
    if not state.settings.enable_smart_prioritization:
        logger.info("Smart prioritization disabled in settings")
        # Pass all failing_functions through unchanged
        return state
    
    logger.info("=" * 70)
    logger.info("SMART PRIORITIZATION (Intelligence Layer)")
    logger.info("=" * 70)
    
    if not state.failing_functions:
        logger.info("No failing functions to prioritize")
        return state
    
    logger.info(f"Analyzing {len(state.failing_functions)} failing targets")
    
    # Match patterns to targets
    prioritized = []
    
    for failing_func in state.failing_functions:
        target_name = failing_func['name']
        module_path = failing_func['module']
        
        # Find patterns for this target
        target_patterns = []
        for pattern in state.coverage_patterns:
            # Match by module and target name
            if pattern.module_path == module_path:
                # Pattern could be for the whole module or specific target
                if target_name in pattern.target_name or pattern.target_name in target_name:
                    target_patterns.append(pattern)
        
        if not target_patterns:
            logger.debug(f"No patterns found for {target_name}, using generic scoring")
        
        # Calculate priority score
        priority_score = _calculate_priority_score(failing_func, target_patterns, state)
        
        # Calculate potential gain
        current_coverage = failing_func['coverage']
        uncovered = len(failing_func['uncovered_lines'])
        potential_gain = (100 - current_coverage) * 0.6  # Estimate 60% of gap can be closed
        
        # Generate reasoning
        reason = _generate_prioritization_reason(failing_func, target_patterns, priority_score)
        
        prioritized_target = PrioritizedTarget(
            target_name=target_name,
            module_path=module_path,
            priority_score=priority_score,
            patterns=target_patterns,
            current_coverage=current_coverage,
            potential_gain=potential_gain,
            reason=reason,
        )
        
        prioritized.append(prioritized_target)
    
    # Sort by priority score (descending)
    prioritized.sort(key=lambda x: x.priority_score, reverse=True)
    
    # Select top N targets
    max_targets = state.settings.max_prioritized_targets
    selected = prioritized[:max_targets]
    skipped = prioritized[max_targets:]
    
    # Update state
    state.prioritized_targets = selected
    state.skipped_targets = [t.target_name for t in skipped]
    
    # Log results
    logger.info("=" * 70)
    logger.info(f"Prioritization Complete")
    logger.info(f"  Total targets analyzed: {len(prioritized)}")
    logger.info(f"  Selected for test generation: {len(selected)}")
    logger.info(f"  Skipped (low ROI): {len(skipped)}")
    logger.info("")
    
    logger.info("Top Prioritized Targets:")
    for idx, target in enumerate(selected[:5], 1):  # Show top 5
        logger.info(f"  {idx}. {target.target_name}")
        logger.info(f"     Score: {target.priority_score:.1f}/10")
        logger.info(f"     Coverage: {target.current_coverage:.1f}%")
        logger.info(f"     Potential gain: +{target.potential_gain:.1f}%")
        logger.info(f"     Reason: {target.reason}")
        logger.info(f"     Patterns: {len(target.patterns)}")
    
    if len(selected) > 5:
        logger.info(f"  ... and {len(selected) - 5} more")
    
    if skipped:
        logger.info("")
        logger.info(f"Skipped Targets (showing first 3):")
        for target in skipped[:3]:
            logger.info(f"  • {target.target_name} (score: {target.priority_score:.1f})")
    
    logger.info("=" * 70)
    
    return state
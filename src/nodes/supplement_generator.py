import logging
from typing import Dict, List
from schema import AgentState
from llm import GroqChatLLM
from prompts_template import USER_SUPPLEMENT_TEMPLATE, SYSTEM_SUPPLEMENT_TESTS

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)



def _highlight_uncovered_lines(source_code: str, uncovered_lines: List[int]) -> str:
    """Add markers to uncovered lines in source code."""
    lines = source_code.split('\n')
    highlighted = []
    
    for i, line in enumerate(lines, 1):
        marker = "  # ⚠️ UNCOVERED" if i in uncovered_lines else ""
        highlighted.append(f"{i:3d} | {line}{marker}")
    
    return '\n'.join(highlighted)


def _analyze_uncovered_lines(source_code: str, uncovered_lines: List[int]) -> List[str]:
    """Provide analysis of what each uncovered line does."""
    lines = source_code.split('\n')
    suggestions = []
    
    for line_no in uncovered_lines:
        if line_no > len(lines):
            continue
        
        line_text = lines[line_no - 1].strip()
        
        if 'raise' in line_text.lower():
            suggestions.append(f"Line {line_no}: Exception path - test should trigger this exception")
        elif 'return' in line_text.lower():
            suggestions.append(f"Line {line_no}: Return statement - test should reach this path")
        elif 'else:' in line_text or line_text.startswith('elif '):
            suggestions.append(f"Line {line_no}: Alternate branch - test different condition")
        elif 'except' in line_text.lower():
            suggestions.append(f"Line {line_no}: Error handler - mock/trigger this exception")
        elif line_text.startswith('if '):
            suggestions.append(f"Line {line_no}: Conditional branch - test when condition is True")
        elif 'for ' in line_text or 'while ' in line_text:
            suggestions.append(f"Line {line_no}: Loop - test with appropriate iterable/condition")
        else:
            suggestions.append(f"Line {line_no}: Statement - ensure execution path reaches here")
    
    return suggestions


def _build_supplement_prompt(failing_func: Dict) -> str:
    """Build targeted prompt for supplemental test generation."""
    func_name = failing_func['name']
    is_method = '.' in func_name
    
    # Highlight uncovered lines
    highlighted = _highlight_uncovered_lines(
        failing_func['source'], 
        failing_func['uncovered_lines']
    )
    
    # Analyze uncovered lines
    suggestions = _analyze_uncovered_lines(
        failing_func['source'],
        failing_func['uncovered_lines']
    )
    
    line_suggestions = '\n'.join(f"  • {s}" for s in suggestions)
    uncovered_str = ', '.join(str(line) for line in failing_func['uncovered_lines'])
    
    # Build method-specific note
    if is_method:
        class_name, method_name = func_name.split('.', 1)
        method_note = f"""IMPORTANT: This is a CLASS METHOD.
- Instantiate the class first: obj = {class_name}(...)
- Mock dependencies in __init__ if needed
- Then call: obj.{method_name}(...)
"""
        import_name = class_name
    else:
        method_note = ""
        import_name = func_name
    
    prompt = USER_SUPPLEMENT_TEMPLATE.format(
        function_name=func_name,
        module_path=failing_func['module'],
        coverage=failing_func['coverage'],
        method_note=method_note,
        highlighted_source=highlighted,
        existing_tests=failing_func['existing_tests'],
        line_suggestions=line_suggestions,
        uncovered_lines=uncovered_str,
        import_name=import_name,
    )
    
    return prompt


def _clean_test_code(raw_response: str) -> str:
    """Clean LLM response to extract pure test code."""
    code = raw_response.strip()
    
    if code.startswith('```python'):
        code = code[len('```python'):].strip()
    elif code.startswith('```'):
        code = code[len('```'):].strip()
    
    if code.endswith('```'):
        code = code[:-3].strip()
    
    lines = code.split('\n')
    first_def_idx = None
    
    for i, line in enumerate(lines):
        if line.strip().startswith('def test_'):
            first_def_idx = i
            break
    
    if first_def_idx is not None:
        code = '\n'.join(lines[first_def_idx:])
    
    return code.strip()


def supplement_tests_node(state: AgentState) -> AgentState:
    """Generate supplemental tests for functions/methods with coverage gaps."""
    logger.info("=" * 60)
    logger.info("Starting Supplemental Test Generation")
    logger.info("=" * 60)
    
    if not state.failing_functions:
        logger.info("No failing functions to supplement")
        state.supplemental_tests = {}
        return state
    
    logger.info(f"Generating supplemental tests for {len(state.failing_functions)} targets")
    
    llm = GroqChatLLM()
    supplemental_tests = {}
    
    for idx, failing_func in enumerate(state.failing_functions, 1):
        func_name = failing_func['name']
        module_path = failing_func['module']
        
        logger.info(f"[{idx}/{len(state.failing_functions)}] Generating tests for {func_name}")
        logger.debug(f"  Module: {module_path}")
        logger.debug(f"  Uncovered lines: {failing_func['uncovered_lines']}")
        
        try:
            user_prompt = _build_supplement_prompt(failing_func)
            
            logger.debug(f"  Prompt size: {len(user_prompt):,} chars")
            
            response = llm.chat(
                system_prompt=SYSTEM_SUPPLEMENT_TESTS,
                user_prompt=user_prompt
            )
            
            logger.debug(f"  Response size: {len(response):,} chars")
            
            cleaned = _clean_test_code(response)
            
            if not cleaned or 'def test_' not in cleaned:
                logger.warning(f"  ⚠️  No valid tests generated for {func_name}")
                continue
            
            test_count = cleaned.count('def test_')
            logger.info(f"  ✓ Generated {test_count} supplemental tests for {func_name}")
            
            if module_path not in supplemental_tests:
                supplemental_tests[module_path] = []
            
            supplemental_tests[module_path].append(cleaned)
            
        except Exception as e:
            logger.error(f"  ✗ Failed to generate tests for {func_name}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            continue
    
    state.supplemental_tests = supplemental_tests
    
    total_modules = len(supplemental_tests)
    total_tests = sum(tests.count('def test_') for tests in 
                     ['\n'.join(t) for t in supplemental_tests.values()])
    
    logger.info("=" * 60)
    logger.info(f"Supplemental Test Generation Complete")
    logger.info(f"  Modules with new tests: {total_modules}")
    logger.info(f"  Total new test functions: {total_tests}")
    logger.info("=" * 60)
    
    return state

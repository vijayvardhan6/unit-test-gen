"""
Test Appender - Appends supplemental tests to existing test files.

This node:
1. For each module with supplemental tests
2. Finds the corresponding test file
3. Appends new tests with iteration marker
4. Preserves all original tests
5. Adds necessary imports if missing
"""

import logging
import os
import re
from typing import Set, Tuple
from schema import AgentState

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _find_test_file(module_path: str, test_root: str) -> str:
    """
    Find the test file path for a given module.
    
    Example: src.data.math -> tests/test_src_data_math.py
    """
    # Normalize module path to filename
    base = module_path.replace('.', '_').replace(':', '_').replace(os.sep, '_')
    if base.endswith('_py'):
        base = base[:-3]
    
    fname = f"test_{base}.py"
    test_file = os.path.join(os.getcwd(), test_root, fname)
    
    return test_file


def _extract_imports_from_tests(test_code: str) -> Set[str]:
    """
    Extract all import statements from test code.
    
    Returns set of import lines.
    """
    imports = set()
    
    for line in test_code.split('\n'):
        stripped = line.strip()
        if stripped.startswith(('import ', 'from ')):
            imports.add(stripped)
    
    return imports


def _read_existing_imports(test_file: str) -> Set[str]:
    """
    Read existing imports from test file.
    """
    if not os.path.exists(test_file):
        return set()
    
    try:
        with open(test_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return _extract_imports_from_tests(content)
    except Exception as e:
        logger.error(f"Failed to read existing imports: {e}")
        return set()


def _get_missing_imports(supplemental_tests: str, existing_imports: Set[str]) -> Set[str]:
    """
    Determine which imports are needed but not present.
    """
    needed_imports = _extract_imports_from_tests(supplemental_tests)
    missing = needed_imports - existing_imports
    
    return missing


def _remove_imports_from_tests(test_code: str) -> Tuple[str, Set[str]]:
    """
    Remove import statements from test code, return both.
    
    Returns: (test_code_without_imports, imports)
    """
    imports = set()
    test_lines = []
    
    for line in test_code.split('\n'):
        stripped = line.strip()
        if stripped.startswith(('import ', 'from ')):
            imports.add(stripped)
        else:
            test_lines.append(line)
    
    # Remove leading empty lines
    while test_lines and not test_lines[0].strip():
        test_lines.pop(0)
    
    return '\n'.join(test_lines), imports


def append_tests_node(state: AgentState) -> AgentState:
    """
    Append supplemental tests to existing test files.
    
    For each module with supplemental tests:
    1. Find test file path
    2. Extract and merge imports
    3. Append tests with iteration marker
    4. Preserve original tests
    """
    logger.info("=" * 60)
    logger.info("Starting Test Appending")
    logger.info("=" * 60)
    
    if not state.supplemental_tests:
        logger.info("No supplemental tests to append")
        return state
    
    logger.info(f"Appending tests for {len(state.supplemental_tests)} modules")
    
    test_root = state.settings.test_root
    iteration = state.iteration_count
    
    for module_path, test_list in state.supplemental_tests.items():
        logger.info(f"Processing module: {module_path}")
        
        # Find test file
        test_file = _find_test_file(module_path, test_root)
        
        if not os.path.exists(test_file):
            logger.warning(f"  ⚠️  Test file not found: {test_file}")
            logger.warning(f"  Creating new test file")
            
            # Create test file with all content
            combined_tests = '\n\n'.join(test_list)
            
            try:
                os.makedirs(os.path.dirname(test_file), exist_ok=True)
                with open(test_file, 'w', encoding='utf-8') as f:
                    f.write(f"# Generated tests - Iteration {iteration}\n\n")
                    f.write(combined_tests)
                    f.write("\n")
                
                logger.info(f"  ✓ Created new test file: {test_file}")
            except Exception as e:
                logger.error(f"  ✗ Failed to create test file: {e}")
                continue
        else:
            # Append to existing file
            logger.info(f"  Appending to: {test_file}")
            
            # Read existing imports
            existing_imports = _read_existing_imports(test_file)
            logger.debug(f"  Found {len(existing_imports)} existing imports")
            
            # Combine all supplemental tests for this module
            combined_tests = '\n\n'.join(test_list)
            
            # Remove imports from supplemental tests
            tests_without_imports, new_imports = _remove_imports_from_tests(combined_tests)
            
            # Find missing imports
            missing_imports = new_imports - existing_imports
            
            try:
                with open(test_file, 'a', encoding='utf-8') as f:
                    # Add section marker
                    f.write("\n\n")
                    f.write("# " + "=" * 70 + "\n")
                    f.write(f"# Iteration {iteration} - Gap Coverage Supplemental Tests\n")
                    f.write("# " + "=" * 70 + "\n\n")
                    
                    # Add missing imports if any
                    if missing_imports:
                        f.write("# Additional imports for supplemental tests\n")
                        for imp in sorted(missing_imports):
                            f.write(f"{imp}\n")
                        f.write("\n")
                        logger.debug(f"  Added {len(missing_imports)} new imports")
                    
                    # Append test functions
                    f.write(tests_without_imports)
                    f.write("\n")
                
                # Count appended tests
                test_count = tests_without_imports.count('def test_')
                logger.info(f"  ✓ Appended {test_count} test functions")
                
            except Exception as e:
                logger.error(f"  ✗ Failed to append tests: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                continue
        
        # Update state.test_file to the last modified file
        state.test_file = test_file
    
    logger.info("=" * 60)
    logger.info(f"Test Appending Complete")
    logger.info(f"  Modified/created {len(state.supplemental_tests)} test files")
    logger.info("=" * 60)
    
    return state
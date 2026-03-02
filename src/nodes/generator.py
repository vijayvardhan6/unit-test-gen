import logging
import re
from typing import Dict, List, Tuple
from schema import AgentState, SourceTarget
from llm import GroqChatLLM
from prompts_template import (
    SYSTEM_TEST_GENERATOR,
    SYSTEM_TEST_GENERATOR_OOP,
    GENERATOR_USER_TEMPLATE_BATCH,
    GENERATOR_USER_TEMPLATE_OOP,
    GENERATOR_USER_TEMPLATE_OOP_BATCH,  
    GENERATOR_USER_TEMPLATE_PER_TARGET,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

BATCH_SIZE = 10  # For functional targets
OOP_BATCH_SIZE = 10  # For OOP targets (slightly smaller due to larger context)


def _create_batches(targets: List[SourceTarget], batch_size: int) -> List[List[SourceTarget]]:
    """Split targets into batches of specified size."""
    batches = []
    for i in range(0, len(targets), batch_size):
        batch = targets[i:i + batch_size]
        batches.append(batch)
    return batches


def _build_batch_context(batch: List[SourceTarget], state: AgentState) -> str:
    """Build combined context for a batch of targets."""
    batch_contexts = []
    
    for idx, target in enumerate(batch, 1):
        context = state.target_contexts.get(target.qualified_name, "")
        
        if not context:
            logger.warning(f"No context found for {target.qualified_name}")
            continue
        
        section = f"""
                {'===================================================='}
                TARGET {idx}/{len(batch)}: {target.symbol}
                Module: {target.module_path}
                Type: {target.target_type.upper()}
                {'===================================================='}

                {context}
                """
        batch_contexts.append(section)
    
    return "\n".join(batch_contexts)


def _build_import_for_target(target: SourceTarget) -> str:
    """
    UNIFIED import builder for any target type.
    """
    if target.target_type == "method":
        return f"from {target.module_path} import {target.parent_class}"
    elif target.target_type == "class":
        return f"from {target.module_path} import {target.symbol}"
    else:  # "function"
        return f"from {target.module_path} import {target.symbol}"


def _build_import_statements(batch: List[SourceTarget]) -> str:
    """Generate deduplicated imports for batch."""
    imports = set()
    
    for target in batch:
        import_line = _build_import_for_target(target)
        imports.add(import_line)
    
    return '\n'.join(sorted(imports))


def _build_oop_batch_prompt(batch: List[SourceTarget], state: AgentState) -> str:
    """
    Build batch prompt for OOP targets using template.
    """
    batch_context = _build_batch_context(batch, state)
    import_statements = _build_import_statements(batch)
    
    # Build target list with type info
    target_list = []
    for idx, target in enumerate(batch, 1):
        if target.target_type == "class":
            target_list.append(f"{idx}. {target.symbol} (CLASS - integration test)")
        elif target.target_type == "method":
            class_name, method_name = target.symbol.split('.', 1)
            target_list.append(f"{idx}. {class_name}.{method_name} (METHOD)")
    
    target_list_str = "\n".join(target_list)
    
    
    return GENERATOR_USER_TEMPLATE_OOP_BATCH.format(
        num_targets=len(batch),
        target_list=target_list_str,
        batch_context=batch_context,
        import_statements=import_statements,
    )

def _is_valid_import_line(line: str) -> bool:
    """
    Validate if an import line is syntactically correct.
    
    Returns False for:
    - Incomplete imports: "from x import"
    - Syntax errors: "from x import {"
    - Empty imports: "import"
    """
    stripped = line.strip()
    
    # Must start with import or from
    if not (stripped.startswith('import ') or stripped.startswith('from ')):
        return False
    
    # Check for incomplete patterns
    invalid_patterns = [
        'import {',
        'import }',
        'import (',
        'import )',
        'import [',
        'import ]',
    ]
    
    for pattern in invalid_patterns:
        if pattern in stripped:
            return False
    
    # "from X import" with nothing after
    if stripped.startswith('from ') and stripped.endswith('import'):
        return False
    
    # "import" with nothing after
    if stripped == 'import' or stripped == 'from':
        return False
    
    # Check for valid import statement structure
    if stripped.startswith('import '):
        # Simple import: "import os" or "import os.path"
        module = stripped.replace('import ', '').strip()
        if not module or not module[0].isalpha():
            return False
    
    elif stripped.startswith('from '):
        # From import: "from x import y"
        if ' import ' not in stripped:
            return False
        
        parts = stripped.split(' import ')
        if len(parts) != 2:
            return False
        
        module = parts[0].replace('from ', '').strip()
        imports = parts[1].strip()
        
        # Module must exist
        if not module or not module[0].isalpha():
            return False
        
        # Must import something
        if not imports or imports in ['{', '}', '(', ')', '[', ']', ',']:
            return False
    
    return True


def _extract_and_organize_imports(response: str) -> Tuple[List[str], str]:
    """
    Extract ALL imports from LLM response and organize them properly.
    
    ENHANCED with:
    - Validation to skip malformed imports
    - Deduplication to prevent repeated imports
    - Automatic dependency detection (e.g., datetime -> timedelta)
    
    Returns:
        (organized_imports, code_without_imports)
    """
    from collections import defaultdict
    
    lines = response.strip().split('\n')
    
    # Categorize imports
    stdlib_simple = set()  # Changed to set for deduplication
    stdlib_from = defaultdict(set)
    
    third_party_simple = set()  # Changed to set
    third_party_from = defaultdict(set)
    
    local_from = defaultdict(set)
    
    code_lines = []
    in_imports_section = True
    
    # Standard library modules
    STDLIB = {
        'os', 'sys', 'json', 'datetime', 're', 'time', 'logging', 
        'collections', 'itertools', 'functools', 'typing', 'pathlib',
        'unittest', 'io', 'tempfile', 'builtins', 'threading', 'subprocess',
        'math', 'random', 'string', 'copy', 'pickle', 'csv', 'xml', 'email'
    }
    
    # Automatic dependency mapping for common cases
    AUTO_DEPENDENCIES = {
        'datetime': {'timedelta', 'timezone'},  # If datetime used, likely needs these
        'pathlib': {'Path'},
        'typing': {'Dict', 'List', 'Optional', 'Any', 'Tuple', 'Set'},
    }
    
    for line in lines:
        stripped = line.strip()
        
        # Skip empty lines
        if not stripped:
            if not in_imports_section:
                code_lines.append(line)
            continue
        
        # Skip comments
        if stripped.startswith('#'):
            if not in_imports_section:
                code_lines.append(line)
            continue
        
        # Validate import line
        if (stripped.startswith('import ') or stripped.startswith('from ')) and in_imports_section:
            if not _is_valid_import_line(stripped):
                logger.warning(f"Skipping malformed import: {stripped}")
                continue
        
        # Simple imports: import X
        if stripped.startswith('import ') and ' as ' not in stripped:
            if in_imports_section:
                module = stripped.replace('import ', '').strip()
                
                if module.split('.')[0] in STDLIB:
                    stdlib_simple.add(stripped)
                else:
                    third_party_simple.add(stripped)
            continue
        
        # From imports: from X import Y
        if stripped.startswith('from '):
            if in_imports_section:
                try:
                    parts = stripped.split(' import ')
                    if len(parts) == 2:
                        module = parts[0].replace('from ', '').strip()
                        imports = parts[1].strip()
                        
                        if not imports or imports in ['{', '}', '(', ')', '[', ']']:
                            logger.warning(f"Skipping incomplete import: {stripped}")
                            continue
                        
                        imports = imports.replace('{', '').replace('}', '').replace('(', '').replace(')', '')
                        imported_items = [item.strip() for item in imports.split(',') if item.strip()]
                        
                        if not imported_items:
                            logger.warning(f"No valid items in import: {stripped}")
                            continue
                        
                        # Categorize
                        if module.split('.')[0] in STDLIB or module.startswith('unittest'):
                            for item in imported_items:
                                if item:
                                    stdlib_from[module].add(item)
                        elif module.startswith(('src.', 'app.', 'lib.', 'package.')):
                            for item in imported_items:
                                if item:
                                    local_from[module].add(item)
                        else:
                            for item in imported_items:
                                if item:
                                    third_party_from[module].add(item)
                except Exception as e:
                    logger.warning(f"Failed to parse import: {stripped} - {e}")
            continue
        
        # Non-import line
        if stripped and not stripped.startswith(('import ', 'from ')):
            in_imports_section = False
        
        if not in_imports_section:
            code_lines.append(line)
    
    # === NEW: Auto-detect missing dependencies in code ===
    code_text = '\n'.join(code_lines)
    
    # Check for common missing imports
    if 'timedelta' in code_text and 'timedelta' not in stdlib_from.get('datetime', set()):
        logger.info("Auto-detected timedelta usage, adding to datetime imports")
        stdlib_from['datetime'].add('timedelta')
    
    if 'timezone' in code_text and 'timezone' not in stdlib_from.get('datetime', set()):
        logger.info("Auto-detected timezone usage, adding to datetime imports")
        stdlib_from['datetime'].add('timezone')
    
    if 'Path' in code_text and 'Path' not in stdlib_from.get('pathlib', set()):
        logger.info("Auto-detected Path usage, adding to pathlib imports")
        stdlib_from['pathlib'].add('Path')
    
    # Build organized import list
    organized = []
    
    # 1. Standard library - simple imports
    if stdlib_simple:
        organized.extend(sorted(stdlib_simple))
    
    # 2. Standard library - from imports (grouped by module)
    if stdlib_from:
        for module in sorted(stdlib_from.keys()):
            items = sorted(stdlib_from[module])
            if len(items) == 1:
                organized.append(f"from {module} import {items[0]}")
            elif len(items) <= 3:
                organized.append(f"from {module} import {', '.join(items)}")
            else:
                organized.append(f"from {module} import (")
                for i, item in enumerate(items):
                    if i < len(items) - 1:
                        organized.append(f"    {item},")
                    else:
                        organized.append(f"    {item}")
                organized.append(")")
    
    if stdlib_simple or stdlib_from:
        organized.append('')
    
    # 3. Third-party - simple imports (deduplicated via set)
    if third_party_simple:
        organized.extend(sorted(third_party_simple))
    
    # 4. Third-party - from imports (grouped)
    if third_party_from:
        for module in sorted(third_party_from.keys()):
            items = sorted(third_party_from[module])
            if len(items) == 1:
                organized.append(f"from {module} import {items[0]}")
            elif len(items) <= 3:
                organized.append(f"from {module} import {', '.join(items)}")
            else:
                organized.append(f"from {module} import (")
                for i, item in enumerate(items):
                    if i < len(items) - 1:
                        organized.append(f"    {item},")
                    else:
                        organized.append(f"    {item}")
                organized.append(")")
    
    if third_party_simple or third_party_from:
        organized.append('')
    
    # 5. Local imports (grouped by module)
    if local_from:
        for module in sorted(local_from.keys()):
            items = sorted(local_from[module])
            if len(items) == 1:
                organized.append(f"from {module} import {items[0]}")
            elif len(items) <= 3:
                organized.append(f"from {module} import {', '.join(items)}")
            else:
                organized.append(f"from {module} import (")
                for i, item in enumerate(items):
                    if i < len(items) - 1:
                        organized.append(f"    {item},")
                    else:
                        organized.append(f"    {item}")
                organized.append(")")
    
    # Remove trailing empty lines
    while organized and organized[-1] == '':
        organized.pop()
    
    code_without_imports = '\n'.join(code_lines).strip()
    
    return organized, code_without_imports


def _parse_batch_response_smart(response: str, batch: List[SourceTarget]) -> Dict[str, str]:
    """
    Parse LLM response with SMART import extraction and deduplication.
    
    FIXED: Prevent duplicate imports by checking if target imports already exist.
    """
    # First, extract and organize ALL imports
    organized_imports, code_without_imports = _extract_and_organize_imports(response)
    
    # Find all test functions
    test_pattern = r'(def test_\w+\([^)]*\):.*?)(?=\ndef test_|\Z)'
    test_matches = re.findall(test_pattern, code_without_imports, re.DOTALL)
    
    if not test_matches:
        logger.error("No test functions found in response!")
        logger.debug(f"Response: {response[:500]}")
        return {}
    
    if len(test_matches) != len(batch):
        logger.warning(f"Expected {len(batch)} tests, got {len(test_matches)}")
    
    # Group tests by module
    tests_by_module = {}
    
    for target, test_code in zip(batch, test_matches):
        module_path = target.module_path
        
        if module_path not in tests_by_module:
            tests_by_module[module_path] = {
                'tests': [],
                'target_imports': set()  # Track target imports per module
            }
        
        tests_by_module[module_path]['tests'].append(test_code)
        
        # Collect target import but don't add yet
        target_import = _build_import_for_target(target)
        tests_by_module[module_path]['target_imports'].add(target_import)
    
    # Build final test files
    result = {}
    
    for module_path, data in tests_by_module.items():
        # Check if target imports already exist in organized imports
        imports_text = '\n'.join(organized_imports)
        missing_target_imports = []
        
        for target_import in sorted(data['target_imports']):
            # Extract the actual import content (handle both styles)
            # Example: "from src.data.main import AlertRule"
            if target_import.startswith('from '):
                # Parse: from X import Y
                parts = target_import.split(' import ')
                if len(parts) == 2:
                    import_module = parts[0].replace('from ', '').strip()
                    import_name = parts[1].strip()
                    
                    # Check if this import already exists
                    # Pattern 1: from src.data.main import AlertRule
                    # Pattern 2: from src.data.main import (AlertRule, ...)
                    if f"from {import_module} import" in imports_text:
                        # Check if the specific name is imported
                        if import_name in imports_text:
                            logger.debug(f"Skipping duplicate import: {target_import}")
                            continue
            
            # This import is not a duplicate
            missing_target_imports.append(target_import)
        
        # Build final imports section
        if missing_target_imports:
            all_imports = organized_imports + [''] + missing_target_imports
            logger.info(f"Module {module_path}: Added {len(missing_target_imports)} target imports "
                       f"(skipped {len(data['target_imports']) - len(missing_target_imports)} duplicates)")
        else:
            all_imports = organized_imports
            logger.info(f"Module {module_path}: All target imports already present, no duplicates added")
        
        imports_section = '\n'.join(all_imports)
        tests_section = '\n\n'.join(data['tests'])
        
        result[module_path] = f"{imports_section}\n\n\n{tests_section}"
        
        logger.info(f"Module {module_path}: {len(data['tests'])} tests, "
                   f"{len(all_imports)} total import lines")
    
    return result


def _generate_single_target(target: SourceTarget, state: AgentState, llm: GroqChatLLM) -> str:
    """Fallback: Generate test for a single target."""
    context = state.target_contexts.get(target.qualified_name, "")
    if not context:
        return ""
    
    # Select appropriate system prompt
    if target.target_type in ["class", "method"]:
        system_prompt = SYSTEM_TEST_GENERATOR_OOP
        
        is_method = target.target_type == "method"
        
        if is_method:
            class_name, method_name = target.symbol.split('.', 1)
            import_name = class_name
            test_name = f"test_{class_name}_{method_name}"
            
            focus_instruction = f"""FOCUS: Test the method {class_name}.{method_name}
- Instantiate {class_name} properly
- Call: obj.{method_name}(...)
- Test this specific method's behavior
"""
        else:
            class_name = target.symbol
            import_name = class_name
            test_name = f"test_{class_name}_integration"
            
            focus_instruction = f"""FOCUS: Test the class {class_name} as an integrated unit
- Test multiple methods working together
- Test realistic workflows
- Verify state changes across method calls
"""
        
        # Build init instruction
        if target.class_init_signature:
            init_instruction = "The class requires initialization. See __init__ in the class definition. Mock dependencies if needed."
        else:
            init_instruction = f"The class has no __init__, instantiate with: obj = {class_name}()"
        
        user_prompt = GENERATOR_USER_TEMPLATE_OOP.format(
            symbol=target.symbol,
            module_path=target.module_path,
            target_type=target.target_type.upper(),
            context=context,
            focus_instruction=focus_instruction,
            import_name=import_name,
            init_instruction=init_instruction,
            test_name=test_name,
        )
    else:
        system_prompt = SYSTEM_TEST_GENERATOR
        user_prompt = GENERATOR_USER_TEMPLATE_PER_TARGET.format(
            target_name=target.symbol,
            module_path=target.module_path,
            qualified_name=target.qualified_name,
            context=context
        )
    
    test_code = llm.chat(
        system_prompt=system_prompt,
        user_prompt=user_prompt
    )
    
    return test_code


def generate_node(state: AgentState) -> AgentState:
    """
    Generate tests with SMART import detection.
    """
    logger.info("Starting test generation with smart import detection")
    
    if not state.targets:
        logger.info("No targets found, skipping generation")
        return state

    has_context_map = hasattr(state, 'target_contexts') and state.target_contexts
    
    if not has_context_map:
        logger.warning("No context map found - falling back")
        return _generate_fallback(state)
    
    # Separate functional and OOP targets
    functional_targets = [t for t in state.targets if t.target_type == "function"]
    oop_targets = [t for t in state.targets if t.target_type in ["class", "method"]]
    
    logger.info(f"Total targets: {len(state.targets)}")
    logger.info(f"  Functional: {len(functional_targets)}")
    logger.info(f"  OOP (classes/methods): {len(oop_targets)}")
    
    llm = GroqChatLLM()
    all_tests = {}  # ← This accumulates across batches
    
    # ==================== PROCESS FUNCTIONAL TARGETS IN BATCHES ====================
    if functional_targets:
        logger.info(f"Processing {len(functional_targets)} functional targets in batches...")
        batches = _create_batches(functional_targets, BATCH_SIZE)
        logger.info(f"Created {len(batches)} functional batches (size: {BATCH_SIZE})")
        
        for batch_idx, batch in enumerate(batches, 1):
            logger.info(f"Functional batch {batch_idx}/{len(batches)} ({len(batch)} targets)")
            
            batch_context = _build_batch_context(batch, state)
            
            if not batch_context.strip():
                logger.warning(f"Empty context for batch {batch_idx}, skipping")
                continue
            
            target_list = "\n".join(
                f"{idx}. {target.symbol} (from {target.module_path})"
                for idx, target in enumerate(batch, 1)
            )
            
            user_prompt = GENERATOR_USER_TEMPLATE_BATCH.format(
                num_targets=len(batch),
                batch_number=batch_idx,
                total_batches=len(batches),
                target_list=target_list,
                batch_context=batch_context,
            )
            
            try:
                response = llm.chat(
                    system_prompt=SYSTEM_TEST_GENERATOR,
                    user_prompt=user_prompt
                )
                
                logger.info(f"Received response for batch {batch_idx} ({len(response):,} chars)")
                
                # Parse batch response
                batch_tests = _parse_batch_response_smart(response, batch)
                
                # ✅ FIXED: Accumulate instead of overwriting
                for module_path, test_code in batch_tests.items():
                    if module_path in all_tests:
                        # Append to existing tests for this module
                        all_tests[module_path] += "\n\n" + test_code
                        logger.debug(f"Appended batch {batch_idx} tests to existing {module_path}")
                    else:
                        # First batch for this module
                        all_tests[module_path] = test_code
                        logger.debug(f"Created new entry for {module_path}")
                
                logger.info(f"Successfully processed batch {batch_idx}")
                
            except Exception as e:
                logger.error(f"Failed to generate batch {batch_idx}: {e}")
                logger.warning(f"Falling back to individual generation")
                
                for target in batch:
                    try:
                        single_test = _generate_single_target(target, state, llm)
                        if single_test:
                            module_path = target.module_path
                            if module_path not in all_tests:
                                all_tests[module_path] = ""
                            all_tests[module_path] += "\n\n" + single_test
                    except Exception as e2:
                        logger.error(f"Failed individual generation for {target.symbol}: {e2}")
    
    # ==================== PROCESS OOP TARGETS (same pattern) ====================
    if oop_targets:
        logger.info(f"Processing {len(oop_targets)} OOP targets in batches...")
        
        oop_batches = _create_batches(oop_targets, OOP_BATCH_SIZE)
        logger.info(f"Created {len(oop_batches)} OOP batches (size: {OOP_BATCH_SIZE})")
        
        for batch_idx, batch in enumerate(oop_batches, 1):
            logger.info(f"OOP batch {batch_idx}/{len(oop_batches)} ({len(batch)} targets)")
            
            try:
                user_prompt = _build_oop_batch_prompt(batch, state)
                
                logger.debug(f"  Prompt size: {len(user_prompt):,} chars")
                
                response = llm.chat(
                    system_prompt=SYSTEM_TEST_GENERATOR_OOP,
                    user_prompt=user_prompt
                )
                
                logger.info(f"Received OOP response for batch {batch_idx} ({len(response):,} chars)")
                
                batch_tests = _parse_batch_response_smart(response, batch)
                
                # ✅ FIXED: Accumulate OOP tests too
                for module_path, test_code in batch_tests.items():
                    if module_path in all_tests:
                        all_tests[module_path] += "\n\n" + test_code
                        logger.debug(f"Appended OOP batch {batch_idx} to {module_path}")
                    else:
                        all_tests[module_path] = test_code
                        logger.debug(f"Created OOP entry for {module_path}")
                
                logger.info(f"✓ Generated {len(batch_tests)} test files in OOP batch {batch_idx}")
                
            except Exception as e:
                logger.error(f"Failed to generate OOP batch {batch_idx}: {e}")
                logger.warning(f"Falling back to individual generation for OOP batch")
                
                for target in batch:
                    try:
                        single_test = _generate_single_target(target, state, llm)
                        if single_test:
                            module_path = target.module_path
                            if module_path not in all_tests:
                                all_tests[module_path] = ""
                            all_tests[module_path] += "\n\n" + single_test
                            logger.info(f"  ✓ Generated test for {target.symbol}")
                    except Exception as e2:
                        logger.error(f"  ✗ Failed for {target.symbol}: {e2}")
    
    state.generated_tests = all_tests
    
    logger.info("=" * 60)
    logger.info(f"Test generation complete!")
    logger.info(f"  Test files generated: {len(all_tests)}")
    
    # Log test counts per module
    for module_path, code in all_tests.items():
        test_count = code.count('def test_')
        logger.info(f"    {module_path}: {test_count} tests")
    
    logger.info("=" * 60)
    
    return state


def _generate_fallback(state: AgentState) -> AgentState:
    """Fallback generation strategy."""
    logger.warning("Using fallback generation strategy")
    
    from collections import defaultdict
    from prompts_template import GENERATOR_USER_TEMPLATE
    
    llm = GroqChatLLM()
    
    grouped = defaultdict(list)
    for t in state.targets:
        grouped[t.module_path].append(t)
    
    generated = dict(state.generated_tests)
    
    for module_path, targets in grouped.items():
        logger.info(f"Generating tests for module: {module_path} (fallback mode)")
        
        source_code = state.source_map.get(module_path, "")
        
        if not source_code:
            logger.warning(f"No source code for {module_path}")
            continue
        
        if len(source_code) > 8000:
            source_code = source_code[:8000] + "\n\n# ... (truncated)"
        
        targets_text = "\n".join(f"- {t.qualified_name}" for t in targets)
        
        user = GENERATOR_USER_TEMPLATE.format(
            module_path=module_path,
            targets_for_module=targets_text,
            source_code=source_code
        )
        
        try:
            test_code = llm.chat(
                system_prompt=SYSTEM_TEST_GENERATOR,
                user_prompt=user
            )
            generated[module_path] = test_code
        except Exception as e:
            logger.error(f"Failed to generate for {module_path}: {e}")
    
    state.generated_tests = generated
    return state
SYSTEM_TEST_GENERATOR = """You are an expert Python test author. Generate clean, modern pytest tests.

CRITICAL OUTPUT REQUIREMENTS:
- Return ONLY valid Python code - NO markdown, NO code fences, NO explanations
- DO NOT wrap code in ```python or ``` 
- Start response directly with 'import' or 'from' statements
- ALL imports must be at the TOP of the file
- End response with the last test function
- No explanatory text before or after the code

CRITICAL ACCURACY REQUIREMENTS:
- STUDY THE ACTUAL FUNCTION SIGNATURES provided in the source code
- DO NOT HALLUCINATE or ASSUME function parameters
- Match EXACT function signatures including parameter names, types, and defaults
- Test what the function ACTUALLY does, not what you think it should do
- ONLY import and use functions that are EXPLICITLY shown in the imports section
- DO NOT use functions from other modules unless they are in the imports

Absolute rules:
- Use pytest only (no unittest).
- Generate EXACTLY ONE test function per target function.
- Include 2-4 assertions per test function to cover different scenarios
- Do NOT use dynamic imports (no __import__, no importlib).
- Do NOT use @pytest.mark.parametrize.
- DO NOT call functions that are not imported

Return ONLY Python code (no explanations).
"""

SYSTEM_TEST_GENERATOR_OOP = """You are an expert Python test author specializing in OOP testing.

CRITICAL OOP TESTING RULES:

1. **For Class Tests**: Test the class as an integrated unit
   - Instantiate the class properly
   - Test method interactions
   - Test state changes across methods

2. **For Method Tests**: Test individual methods with proper setup
   - Create class instance first
   - Mock dependencies in __init__ if needed
   - Test method in isolation

3. **Class Instantiation Patterns**:
   - Read __init__ signature from provided source
   - Mock external dependencies (DB, API, etc.)
   - Use realistic values for simple parameters

4. **Common Patterns**:
   - Dataclasses: Instantiate directly with keyword args
   - Database models: Mock connection/session
   - API clients: Mock requests/httpx
   - Services: Mock dependencies via constructor injection

CRITICAL OUTPUT REQUIREMENTS:
- Return ONLY valid Python code
- NO markdown, NO code fences, NO explanations
- Start with 'import' or 'from' statements
- ALL imports must be at the TOP of the file
- End with the last test function

EXAMPLE - Testing a Method:

# Given class:
class UserService:
    def __init__(self, db_connection):
        self.db = db_connection
    
    def get_user(self, user_id: int) -> dict:
        return self.db.query("SELECT * FROM users WHERE id = ?", user_id)

'========================================'

# Good test:
from unittest.mock import Mock
import pytest
from src.services import UserService

def test_UserService_get_user():
    # Arrange: Mock dependencies
    mock_db = Mock()
    mock_db.query.return_value = {'id': 1, 'name': 'Alice'}
    
    # Act: Instantiate and call method
    service = UserService(mock_db)
    result = service.get_user(1)
    
    # Assert
    assert result['id'] == 1
    assert result['name'] == 'Alice'
    mock_db.query.assert_called_once_with("SELECT * FROM users WHERE id = ?", 1)

Return ONLY Python code (no explanations).
"""

GENERATOR_USER_TEMPLATE_BATCH = """Generate pytest tests for {num_targets} functions.

Batch {batch_number} of {total_batches}

TARGETS TO TEST:
{target_list}

SOURCE CODE:
{batch_context}

IMPORTANT RESTRICTIONS:
- You can ONLY use the functions listed in the imports above
- DO NOT call any other functions
- If a function is not in the imports, you CANNOT use it
- Only test the specific functions you imported
- Use appropriate mocking tools (Mock, patch, MagicMock) as needed

Now generate {num_targets} test functions (one for each imported function).

Each test should:
- Test ONLY the function you imported for that test
- Use Mock() for any database connections or external dependencies
- Have 2-4 assertions covering happy path, edge cases, and errors
- NOT call functions from other modules

Return ONLY the test function code (no explanations).
"""

GENERATOR_USER_TEMPLATE_OOP_BATCH = """Generate pytest tests for {num_targets} OOP targets (classes/methods).

TARGETS TO TEST:
{target_list}

======================================================================
SOURCE CODE WITH FULL CLASS CONTEXT:
======================================================================

{batch_context}

======================================================================
CRITICAL IMPORT REQUIREMENTS:
======================================================================

Your response MUST start with these imports ONCE at the top:

{import_statements}

DO NOT repeat these imports before each test function!

======================================================================
INSTRUCTIONS:
======================================================================

Generate {num_targets} test functions (one per target).

For CLASS targets:
- Test the class as an integrated unit
- Test multiple methods working together
- Name: test_ClassName_integration

For METHOD targets:
- Instantiate the parent class first
- Mock __init__ dependencies if needed
- Test the specific method
- Name: test_ClassName_methodname

Each test should:
- Import what you need (pytest, Mock, patch, datetime, json, etc.)
- Have 2-4 assertions
- Cover happy path and edge cases
- Mock external dependencies appropriately

CRITICAL OUTPUT FORMAT:
1. Imports at the TOP (once)
2. Then all test functions
3. NO imports between test functions
4. NO markdown, NO explanations

Return ONLY Python code.
"""

GENERATOR_USER_TEMPLATE_OOP = """Generate pytest test for: {symbol}
Module: {module_path}
Type: {target_type}

======================================================================
SOURCE CODE:
======================================================================

{context}

======================================================================
INSTRUCTIONS:
======================================================================

{focus_instruction}

Import: from {module_path} import {import_name}

{init_instruction}

Generate ONE test function named {test_name}:
- Include 2-4 assertions
- Cover happy path and edge cases
- Mock external dependencies

Return ONLY test code (no explanations).
"""

GENERATOR_USER_TEMPLATE_PER_TARGET = """Generate a comprehensive pytest test for this specific function.

Target Function: {target_name}
Module: {module_path}
Qualified Name: {qualified_name}

======================================================================
RELEVANT CODE CONTEXT (Includes function + dependencies):
======================================================================

{context}

======================================================================

## CRITICAL INSTRUCTIONS

1. **Study the code above** - It contains:
   - The target function you're testing
   - Any functions it calls (dependencies)
   - Any classes it uses (Pydantic models, etc.)
   - All relevant imports
   
2. **DO NOT GUESS** - Everything you need is in the context above:
   - Exact parameter names and types
   - Default values
   - Return types
   - Exception handling

3. **Generate ONE test function** named test_{target_name} with 2-4 assertions:
   - Happy path: Normal inputs that should work (1-2 assertions)
   - Edge cases: Empty/None/zero/boundaries (1-2 assertions)
   - Error conditions: Invalid inputs with pytest.raises() (0-1 assertion)

4. **For mocking**:
   - Mock ALL external dependencies (requests, database, file I/O)
   - Mock functions must accept ALL parameters including **kwargs
   - Include ALL methods the real object has (e.g., raise_for_status)

5. **Import statement** at the top:
   from {module_path} import {target_name}

**Return ONLY the test function code - no explanations, no markdown.**
"""

GENERATOR_USER_TEMPLATE = """Generate comprehensive pytest tests for the following module.

Module: {module_path}

Targets to test:
{targets_for_module}

======================================================================
ACTUAL SOURCE CODE (READ THIS CAREFULLY):
======================================================================

{source_code}

======================================================================

## CRITICAL REQUIREMENTS

1. **DO NOT GUESS FUNCTION SIGNATURES** - Use the EXACT signatures from the source code above
   
2. **Check the source code for**:
   - Exact parameter names, types, and defaults
   - Return types (Dict vs DataFrame vs List vs ndarray)
   - Whether Pydantic fields are required or optional
   - What exceptions are raised and when
   
3. **Import correctly**: 
   from {module_path} import function_name, ClassName

4. **For each target, create ONE test function with 2-4 assertions**:
   - Happy path (normal inputs): 1-2 assertions
   - Edge cases (boundaries, empty, None, zero): 1-2 assertions  
   - Error conditions (invalid inputs): 0-1 assertion using pytest.raises()

5. **For mocking external dependencies**:
   - Mock ALL methods used by the function
   - Mock functions must accept ALL parameters including **kwargs

REMEMBER: The source code above is the TRUTH. Your tests must match it EXACTLY.
"""

SYSTEM_SUPPLEMENT_TESTS = """You are an expert Python test author specializing in improving code coverage.

Your task is to generate SUPPLEMENTAL tests that cover specific uncovered lines.

CRITICAL RULES:
- Generate ONLY 2-3 NEW test functions
- DO NOT regenerate existing tests
- Focus ONLY on covering the specified uncovered lines
- Each test should target specific branches, edge cases, or error conditions
- Use pytest syntax
- Include 2-4 assertions per test
- Return ONLY Python code - NO markdown, NO explanations, NO code fences

For CLASS METHODS:
- Remember to instantiate the class first
- Mock dependencies in __init__ if needed
- Then call the method: obj.method_name(...)

Output Format:
def test_function_name_edge_case():
    # Test specific uncovered branch
    assert function(edge_input) == expected
    
def test_function_name_error_condition():
    # Test error path
    with pytest.raises(ExpectedException):
        function(invalid_input)
"""

USER_SUPPLEMENT_TEMPLATE = """Function/Method: {function_name}
Module: {module_path}
Current Coverage: {coverage:.1f}%

{method_note}

SOURCE CODE WITH UNCOVERED LINES MARKED:
{highlighted_source}

EXISTING TESTS (DO NOT REGENERATE):
{existing_tests}

UNCOVERED LINES ANALYSIS:
{line_suggestions}

TASK:
Generate 2-3 NEW test functions that specifically cover the uncovered lines above.

Requirements:
- Test different input combinations than existing tests
- Focus on branches, edge cases, and error conditions
- Target lines: {uncovered_lines}
- Use function signature from source code above
- Import statement: from {module_path} import {import_name}

Return ONLY the test function code (no imports, no explanations).
"""
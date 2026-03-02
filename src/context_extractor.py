import ast
import logging
import os
from typing import Dict, List, Set, Optional, Any
from schema import AgentState, SourceTarget

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DependencyAnalyzer(ast.NodeVisitor):
    """AST visitor to find all names referenced in a function/class."""
    
    def __init__(self):
        self.dependencies: Set[str] = set()
        
    def visit_Name(self, node: ast.Name):
        self.dependencies.add(node.id)
        self.generic_visit(node)
        
    def visit_Attribute(self, node: ast.Attribute):
        if isinstance(node.value, ast.Name):
            self.dependencies.add(node.value.id)
        self.generic_visit(node)
        
    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Name):
            self.dependencies.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                self.dependencies.add(node.func.value.id)
        self.generic_visit(node)


def _parse_imports(tree: ast.AST) -> Dict[str, str]:
    """Extract all imports. Returns: {name: module}"""
    imports = {}
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name
                imports[name] = alias.name
                
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name
                imports[name] = f"{module}.{alias.name}" if module else alias.name
    
    return imports


def _find_definition(tree: ast.AST, name: str) -> Optional[ast.AST]:
    """Find function or class definition by name."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            if node.name == name:
                return node
    return None


def _find_class_definition(tree: ast.AST, class_name: str) -> Optional[ast.ClassDef]:
    """Find a class definition by name."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    return None


def _find_method_in_class(class_node: ast.ClassDef, method_name: str) -> Optional[ast.FunctionDef]:
    """Find a method within a class definition."""
    for item in class_node.body:
        if isinstance(item, ast.FunctionDef) and item.name == method_name:
            return item
    return None


def _extract_class_init_info(class_node: ast.ClassDef) -> Dict[str, Any]:
    """
    Extract __init__ signature and parameters.
    """
    for item in class_node.body:
        if isinstance(item, ast.FunctionDef) and item.name == '__init__':
            try:
                signature = ast.unparse(item)
                
                params = []
                for arg in item.args.args:
                    if arg.arg != 'self':
                        params.append(arg.arg)
                
                defaults = item.args.defaults
                required_params = len(params) - len(defaults)
                
                return {
                    'has_init': True,
                    'signature': signature,
                    'params': params,
                    'required_params': params[:required_params],
                    'optional_params': params[required_params:],
                    'source': signature
                }
            except Exception as e:
                logger.warning(f"Failed to extract __init__ info: {e}")
    
    return {
        'has_init': False,
        'signature': None,
        'params': [],
        'required_params': [],
        'optional_params': [],
        'source': None
    }


def _extract_class_attributes(class_node: ast.ClassDef) -> List[str]:
    """Extract class-level attributes."""
    attributes = []
    
    for item in class_node.body:
        if isinstance(item, ast.Assign):
            for target in item.targets:
                if isinstance(target, ast.Name):
                    if not target.id.startswith('_'):
                        attributes.append(target.id)
        
        elif isinstance(item, ast.AnnAssign):
            if isinstance(item.target, ast.Name):
                if not item.target.id.startswith('_'):
                    attributes.append(item.target.id)
    
    return attributes


def _extract_instance_attributes(class_node: ast.ClassDef) -> List[str]:
    """Extract instance attributes set in __init__."""
    init_node = _find_method_in_class(class_node, '__init__')
    if not init_node:
        return []
    
    instance_attrs = []
    
    for node in ast.walk(init_node):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Attribute):
                    if isinstance(target.value, ast.Name) and target.value.id == 'self':
                        if not target.attr.startswith('_'):
                            instance_attrs.append(target.attr)
    
    return instance_attrs


def _is_dataclass(class_node: ast.ClassDef) -> bool:
    """Check if class has @dataclass decorator."""
    for decorator in class_node.decorator_list:
        if isinstance(decorator, ast.Name) and decorator.id == 'dataclass':
            return True
        if isinstance(decorator, ast.Call):
            if isinstance(decorator.func, ast.Name) and decorator.func.id == 'dataclass':
                return True
    return False


def _extract_base_classes(class_node: ast.ClassDef) -> List[str]:
    """Extract base class names."""
    bases = []
    for base in class_node.bases:
        if isinstance(base, ast.Name):
            bases.append(base.id)
        elif isinstance(base, ast.Attribute):
            try:
                bases.append(ast.unparse(base))
            except:
                pass
    return bases


def _get_dependencies_for_target(tree: ast.AST, target_name: str) -> Set[str]:
    """Find all names that a target depends on."""
    target_node = _find_definition(tree, target_name)
    if not target_node:
        return set()
    
    analyzer = DependencyAnalyzer()
    analyzer.visit(target_node)
    return analyzer.dependencies


def _extract_relevant_definitions(tree: ast.AST, dependencies: Set[str]) -> List[ast.AST]:
    """Extract function/class definitions in dependency set."""
    relevant = []
    
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            if node.name in dependencies:
                relevant.append(node)
    
    return relevant


def _extract_relevant_imports(imports: Dict[str, str], dependencies: Set[str]) -> List[str]:
    """Extract only imports that are actually used."""
    relevant_imports = []
    
    for name, module in imports.items():
        if name in dependencies:
            if '.' in module:
                parts = module.rsplit('.', 1)
                if len(parts) == 2:
                    from_module, import_name = parts
                    if name == import_name:
                        relevant_imports.append(f"from {from_module} import {import_name}")
                    else:
                        relevant_imports.append(f"from {from_module} import {import_name} as {name}")
                else:
                    relevant_imports.append(f"import {module}")
            else:
                if name == module:
                    relevant_imports.append(f"import {module}")
                else:
                    relevant_imports.append(f"import {module} as {name}")
    
    return relevant_imports


def extract_oop_context(
    class_name: str,
    tree: ast.AST,
    imports: Dict[str, str],
    focus_method: Optional[str] = None
) -> str:
    """
    UNIFIED context extractor for classes and methods.
    
    Args:
        class_name: Name of the class
        tree: AST of the module
        imports: Import mapping
        focus_method: If provided, highlight this method as target
    
    Returns:
        Formatted context string
    """
    class_node = _find_class_definition(tree, class_name)
    if not class_node:
        logger.warning(f"Class {class_name} not found")
        return f"# Class {class_name} not found"
    
    # Extract class metadata (same for both class and method targets)
    init_info = _extract_class_init_info(class_node)
    class_attrs = _extract_class_attributes(class_node)
    instance_attrs = _extract_instance_attributes(class_node)
    is_dc = _is_dataclass(class_node)
    bases = _extract_base_classes(class_node)
    
    # Build context
    context_parts = []
    
    # Add relevant imports
    dependencies = {class_name}
    relevant_imports = _extract_relevant_imports(imports, dependencies)
    if relevant_imports:
        context_parts.append("# Imports")
        context_parts.extend(relevant_imports)
        context_parts.append("")
    
    # Add metadata
    metadata = []
    
    if focus_method:
        metadata.append(f"# Target: {class_name}.{focus_method} (METHOD)")
        metadata.append(f"# Parent Class: {class_name}")
    else:
        metadata.append(f"# Target: {class_name} (CLASS)")
    
    if bases:
        metadata.append(f"# Inherits from: {', '.join(bases)}")
    
    if is_dc:
        metadata.append("# Note: This is a @dataclass")
    
    if init_info['has_init']:
        required = init_info['required_params']
        optional = init_info['optional_params']
        if required:
            metadata.append(f"# Required __init__ params: {', '.join(required)}")
        if optional:
            metadata.append(f"# Optional __init__ params: {', '.join(optional)}")
    else:
        metadata.append("# No __init__ defined (uses default)")
    
    if instance_attrs:
        metadata.append(f"# Instance attributes: {', '.join(instance_attrs)}")
    
    if class_attrs:
        metadata.append(f"# Class attributes: {', '.join(class_attrs)}")
    
    context_parts.extend(metadata)
    context_parts.append("")
    
    # Add full class source
    context_parts.append("# Full class definition:")
    context_parts.append(ast.unparse(class_node))
    
    # If method specified, add highlighted section
    if focus_method:
        method_node = _find_method_in_class(class_node, focus_method)
        if method_node:
            context_parts.append("")
            context_parts.append("=" * 70)
            context_parts.append(f"# ⭐ TARGET METHOD TO TEST:")
            context_parts.append("=" * 70)
            context_parts.append(ast.unparse(method_node))
        else:
            logger.warning(f"Method {focus_method} not found in {class_name}")
    
    return "\n".join(context_parts)


def extract_function_context(
    function_name: str,
    tree: ast.AST,
    imports: Dict[str, str]
) -> str:
    """Extract context for standalone function."""
    dependencies = _get_dependencies_for_target(tree, function_name)
    relevant_imports = _extract_relevant_imports(imports, dependencies)
    relevant_defs = _extract_relevant_definitions(tree, dependencies)
    target_node = _find_definition(tree, function_name)
    
    context_parts = []
    
    if relevant_imports:
        context_parts.append("# Imports")
        context_parts.extend(relevant_imports)
        context_parts.append("")
    
    if relevant_defs:
        context_parts.append("# Dependent definitions")
        for node in relevant_defs:
            if node.name != function_name:
                context_parts.append(ast.unparse(node))
                context_parts.append("")
    
    if target_node:
        context_parts.append(f"# Target: {function_name}")
        context_parts.append(ast.unparse(target_node))
    else:
        logger.warning(f"Could not find target {function_name}")
        context_parts.append(f"# Target {function_name} not found")
    
    return "\n".join(context_parts)


def extract_context_for_target(
    target: SourceTarget,
    state: AgentState
) -> str:
    """
    UNIFIED context extractor - handles function, class, or method.
    """
    module_path = target.module_path
    source_code = state.source_map.get(module_path, "")
    
    if not source_code:
        logger.warning(f"No source code found for {module_path}")
        return f"# Source code not found for {module_path}"
    
    try:
        tree = ast.parse(source_code)
        imports = _parse_imports(tree)
    except SyntaxError as e:
        logger.error(f"Syntax error parsing {module_path}: {e}")
        return source_code[:2000]
    
    # Unified handling based on target type
    if target.target_type in ["class", "method"]:
        # Both are OOP - differ only in focus
        class_name = target.parent_class if target.target_type == "method" else target.symbol
        focus_method = target.symbol.split('.')[-1] if target.target_type == "method" else None
        
        return extract_oop_context(
            class_name=class_name,
            tree=tree,
            imports=imports,
            focus_method=focus_method
        )
    
    else:  # "function"
        return extract_function_context(
            function_name=target.symbol,
            tree=tree,
            imports=imports
        )


def build_context_map(state: AgentState) -> Dict[str, str]:
    """
    Build a map of target qualified_name -> minimal context.
    """
    context_map = {}
    
    logger.info(f"Building context map for {len(state.targets)} targets")
    
    for target in state.targets:
        context = extract_context_for_target(target, state)
        context_map[target.qualified_name] = context
        
        logger.debug(f"Built context for {target.qualified_name}: {len(context)} chars")
    
    logger.info(f"Context map built with {len(context_map)} entries")
    return context_map
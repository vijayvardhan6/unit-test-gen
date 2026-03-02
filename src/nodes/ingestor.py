import ast
import logging
import os
from typing import List

from schema import AgentState, SourceTarget
from context_extractor import build_context_map

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _is_public(name: str) -> bool:
    return not name.startswith("_")


def _collect_files(paths: List[str], exclude_dirs: List[str]) -> List[str]:
    files = []
    for p in paths:
        if os.path.isfile(p) and p.endswith(".py"):
            files.append(os.path.abspath(p))
        elif os.path.isdir(p):
            for root, dirs, filenames in os.walk(p):
                dirs[:] = [d for d in dirs if d not in exclude_dirs]
                for f in filenames:
                    if f.endswith(".py"):
                        files.append(os.path.abspath(os.path.join(root, f)))
    return files


def _find_project_root(paths: List[str]) -> str:
    if not paths:
        return os.getcwd()
    
    first_path = os.path.abspath(paths[0])
    if os.path.isfile(first_path):
        first_path = os.path.dirname(first_path)
    
    path_parts = first_path.split(os.sep)
    root_indicators = ['src', 'app', 'lib', 'package', 'core', 'modules']
    
    for i, part in enumerate(path_parts):
        if part in root_indicators:
            project_root = os.sep.join(path_parts[:i])
            if project_root:
                logger.info(f"Auto-detected project root: {project_root} (based on '{part}' indicator)")
                return project_root
    
    cwd = os.getcwd()
    logger.info(f"Using current working directory as project root: {cwd}")
    return cwd


def _module_from_path(path: str, import_base: str) -> str:
    """
    Convert file path to Python module path relative to import_base.
    """
    abs_path = os.path.abspath(path)
    abs_base = os.path.abspath(import_base)
    
    try:
        rel_path = os.path.relpath(abs_path, abs_base)
        
        if rel_path.startswith('..'):
            logger.error(f"File {abs_path} is NOT under import base {abs_base}")
            basename = os.path.basename(abs_path)
            if basename.endswith('.py'):
                basename = basename[:-3]
            return basename
        
        mod = rel_path.replace(os.sep, ".")
        if mod.endswith(".py"):
            mod = mod[:-3]
        
        logger.debug(f"Module: {abs_path} -> {mod}")
        return mod
        
    except ValueError as e:
        logger.error(f"Cannot create relative path: {e}")
        basename = os.path.basename(path)
        if basename.endswith(".py"):
            basename = basename[:-3]
        return basename


def _parse_targets(path: str, code: str, project_root: str) -> List[SourceTarget]:
    """
    Parse Python source code.
    """
    targets: List[SourceTarget] = []
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        logger.error(f"Syntax error in {path}: {str(e)}")
        return targets

    module_path = _module_from_path(path, project_root)
    logger.debug(f"Parsing module: {module_path}")

    functions = 0
    classes = 0
    methods = 0
    
    for node in tree.body:
        # Standalone functions
        if isinstance(node, ast.FunctionDef) and _is_public(node.name):
            targets.append(SourceTarget(
                module_path=module_path,
                symbol=node.name,
                qualified_name=f"{module_path}:{node.name}",
                target_type="function",
            ))
            functions += 1
        
        # Classes with full metadata
        if isinstance(node, ast.ClassDef) and _is_public(node.name):
            class_name = node.name
            
            # Extract __init__ info
            init_signature = None
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == '__init__':
                    try:
                        init_signature = ast.unparse(item)
                    except:
                        pass
                    break
            
            # Extract class attributes
            class_attrs = []
            for item in node.body:
                if isinstance(item, ast.Assign):
                    for target in item.targets:
                        if isinstance(target, ast.Name) and not target.id.startswith('_'):
                            class_attrs.append(target.id)
            
            # Check for dataclass
            is_dataclass = any(
                (isinstance(d, ast.Name) and d.id == 'dataclass') or
                (isinstance(d, ast.Call) and isinstance(d.func, ast.Name) and d.func.id == 'dataclass')
                for d in node.decorator_list
            )
            
            # Extract base classes
            bases = []
            for base in node.bases:
                if isinstance(base, ast.Name):
                    bases.append(base.id)
                elif isinstance(base, ast.Attribute):
                    try:
                        bases.append(ast.unparse(base))
                    except:
                        pass
            
            # Extract all public methods
            method_names = []
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and _is_public(item.name):
                    method_names.append(item.name)
            
            # Create class-level target
            targets.append(SourceTarget(
                module_path=module_path,
                symbol=class_name,
                qualified_name=f"{module_path}:{class_name}",
                target_type="class",
                class_methods=method_names,
                class_init_signature=init_signature,
                class_attributes=class_attrs,
                is_dataclass=is_dataclass,
                base_classes=bases,
            ))
            classes += 1
            
            # Create individual method targets
            for method_name in method_names:
                targets.append(SourceTarget(
                    module_path=module_path,
                    symbol=f"{class_name}.{method_name}",
                    qualified_name=f"{module_path}:{class_name}.{method_name}",
                    target_type="method",
                    parent_class=class_name,
                    class_init_signature=init_signature,
                    is_dataclass=is_dataclass,
                    base_classes=bases,
                ))
                methods += 1

    if targets:
        logger.debug(f"Found in {module_path}: {functions} functions, {classes} classes, {methods} methods")
    return targets


def ingest_node(state: AgentState) -> AgentState:
    """
    Collect source files, discover symbols, and build context map.
    """
    logger.info("Starting source code ingestion")
    source_map = dict(state.source_map)
    targets: List[SourceTarget] = []
    source_files_list = []
    
    if state.settings.project_root:
        project_root = state.settings.project_root
        logger.info(f"Using configured project root: {project_root}")
    else:
        project_root = _find_project_root(state.paths)
    
    if not os.path.exists(project_root):
        logger.error(f"Project root does not exist: {project_root}")
        logger.info("Falling back to current working directory")
        project_root = os.getcwd()
    
    logger.info(f"Project root: {project_root}")

    if state.inline_code:
        logger.info("Processing inline code")
        module_path = "inline_module"
        source_map[module_path] = state.inline_code
        targets += _parse_targets("inline_module", state.inline_code, project_root)

    logger.info(f"Collecting Python files from {len(state.paths)} paths")
    files = _collect_files(state.paths, state.settings.exclude_dirs)
    logger.info(f"Found {len(files)} Python files to process")

    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as fh:
                code = fh.read()
            
            module_path = _module_from_path(f, project_root)
            source_map[module_path] = code
            source_files_list.append(f)
            
            file_targets = _parse_targets(f, code, project_root)
            targets += file_targets
            
            logger.info(f"Processed: {os.path.basename(f)} -> module={module_path}, targets={len(file_targets)}")
        except Exception as e:
            logger.error(f"Failed to process file {f}: {str(e)}")
            continue

    state.source_map = source_map
    state.source_files = source_files_list
    state.targets = targets
    
    logger.info(f"Ingestion complete. Processed {len(source_map)} modules, found {len(targets)} targets")
    
    if targets:
        sample_modules = list(set(t.module_path for t in targets[:5]))
        logger.info(f"Sample module paths: {sample_modules}")
        
        sample_qualified = [t.qualified_name for t in targets[:3]]
        logger.info(f"Sample qualified names: {sample_qualified}")
    
    # Build context map
    logger.info("=" * 60)
    logger.info("Building context map for targets")
    logger.info("=" * 60)
    
    try:
        context_map = build_context_map(state)
        state.target_contexts = context_map
        
        total_context_size = sum(len(ctx) for ctx in context_map.values())
        avg_context_size = total_context_size // len(context_map) if context_map else 0
        
        logger.info(f"Context map statistics:")
        logger.info(f"  Total targets: {len(context_map)}")
        logger.info(f"  Total context size: {total_context_size:,} chars")
        logger.info(f"  Average context per target: {avg_context_size:,} chars")
        
        if context_map:
            sample_sizes = [(k, len(v)) for k, v in list(context_map.items())[:3]]
            for name, size in sample_sizes:
                logger.info(f"    {name}: {size:,} chars")
        
    except Exception as e:
        logger.error(f"Failed to build context map: {e}")
        import traceback
        logger.error(traceback.format_exc())
        state.target_contexts = {}
    
    logger.info("=" * 60)
    
    return state
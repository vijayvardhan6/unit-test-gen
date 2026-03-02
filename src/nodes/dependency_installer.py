from __future__ import annotations
import ast
import logging
import os
import subprocess
import sys
from typing import Set, List, Dict, Any
from schema import AgentState

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Comprehensive standard library modules (Python 3.8+)
STDLIB_MODULES = {
    'os', 'sys', 'datetime', 'json', 'logging', 're', 'typing', 'time',
    'collections', 'itertools', 'functools', 'pathlib', 'subprocess',
    'xml', 'ast', 'inspect', 'unittest', 'pytest', 'pdb', 'traceback', 'warnings',
    'argparse', 'configparser', 'io', 'tempfile', 'shutil', 'glob',
    'random', 'math', 'statistics', 'decimal', 'fractions', 'numbers',
    'string', 'copy', 'pickle', 'shelve', 'marshal', 'dbm',
    'sqlite3', 'csv', 'zipfile', 'tarfile', 'gzip', 'bz2', 'lzma',
    'hashlib', 'hmac', 'secrets', 'base64', 'binascii', 'struct',
    'codecs', 'unicodedata', 'locale', 'gettext',
    'threading', 'multiprocessing', 'concurrent', 'queue', 'asyncio',
    'socket', 'ssl', 'select', 'selectors', 'signal',
    'email', 'mimetypes', 'urllib', 'http', 'ftplib', 'smtplib',
    'uuid', 'html', 'webbrowser',
    'dataclasses', 'enum', 'abc', 'contextlib', 'weakref',
    'operator', 'heapq', 'bisect', 'array', 'types',
    'pprint', 'reprlib', 'dis', 'pickletools',
    'platform', 'errno', 'ctypes',
    'importlib', 'pkgutil', 'modulefinder', 'runpy',
    'atexit', 'gc', 'copyreg',
    'typing_extensions',
}

def _extract_imports_from_code(code: str) -> Set[str]:
    """Extract all top-level package imports from Python code."""
    imports = set()
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root_pkg = alias.name.split('.')[0]
                    imports.add(root_pkg)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    root_pkg = node.module.split('.')[0]
                    imports.add(root_pkg)
    except SyntaxError as e:
        logger.warning(f"Syntax error while parsing imports: {e}")
    except Exception as e:
        logger.error(f"Error extracting imports: {e}")
    return imports


def _extract_local_module_names(state: AgentState) -> Set[str]:
    """
    Extract all local module names from source_map.
    
    This is the RELIABLE way - we get actual module names that exist in the project.
    
    Examples:
        source_map keys: ['src.data.flights', 'src.data.basics', 'src.utils.helper']
        returns: {'src', 'data', 'flights', 'basics', 'utils', 'helper'}
    """
    local_modules = set()
    
    for module_path in state.source_map.keys():
        # Split by dots and add all parts
        parts = module_path.split('.')
        local_modules.update(parts)
        
        # Also add the last part (the actual module name) 
        # e.g., for 'src.data.flights' we definitely want 'flights'
        if parts:
            local_modules.add(parts[-1])
    
    logger.debug(f"Extracted {len(local_modules)} local module names from source_map")
    logger.debug(f"Local modules: {sorted(local_modules)}")
    
    return local_modules


def _get_third_party_imports(all_imports: Set[str], state: AgentState) -> Set[str]:
    """
    Filter out standard library modules and local modules.
    """
    # Get actual local module names from source_map
    local_modules = _extract_local_module_names(state)
    
    third_party = set()
    
    for pkg in all_imports:
        # Skip standard library
        if pkg in STDLIB_MODULES:
            logger.debug(f"Skipping '{pkg}' (standard library)")
            continue
        
        # Skip if it's a local module (from source_map)
        if pkg in local_modules:
            logger.debug(f"Skipping '{pkg}' (local module from source_map)")
            continue
        
        # Add to third-party list
        logger.debug(f"'{pkg}' identified as third-party package")
        third_party.add(pkg)
    
    return third_party


def _check_package_installed(package: str) -> bool:
    """Check if a package is already installed."""
    try:
        __import__(package)
        return True
    except ImportError:
        return False


def _install_packages(packages: List[str], timeout: int = 600) -> Dict[str, Any]:
    """Install packages using pip."""
    if not packages:
        return {"status": "skipped", "message": "No packages to install"}
    
    # Filter out already installed packages
    to_install = [pkg for pkg in packages if not _check_package_installed(pkg)]
    
    if not to_install:
        logger.info(f"All packages already installed: {packages}")
        return {
            "status": "success",
            "installed": [],
            "already_installed": packages,
            "message": "All packages already available"
        }
    
    logger.info(f"Installing {len(to_install)} packages: {to_install}")
    
    installed = []
    failed = []
    
    # Install packages one by one for better error tracking
    for pkg in to_install:
        try:
            cmd = [
                sys.executable, "-m", "pip", "install",
                pkg,
                "--quiet",
                "--disable-pip-version-check",
                "--no-warn-script-location"
            ]
            
            logger.debug(f"Installing {pkg}...")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            if result.returncode == 0:
                logger.info(f"✓ Successfully installed: {pkg}")
                installed.append(pkg)
            else:
                logger.error(f"✗ Failed to install {pkg}: {result.stderr}")
                failed.append(pkg)
                
        except subprocess.TimeoutExpired:
            logger.error(f"✗ Timeout installing {pkg}")
            failed.append(pkg)
        except Exception as e:
            logger.error(f"✗ Error installing {pkg}: {e}")
            failed.append(pkg)
    
    if installed:
        logger.info(f"Successfully installed {len(installed)} packages: {', '.join(installed)}")
    
    if failed:
        logger.warning(f"Failed to install {len(failed)} packages: {', '.join(failed)}")
    
    return {
        "status": "success" if not failed else "partial",
        "installed": installed,
        "failed": failed,
        "already_installed": [p for p in packages if p not in to_install]
    }


def dependency_installer_node(state: AgentState) -> AgentState:
    """
    Analyze source code and install required dependencies before test execution.
    """
    logger.info("=" * 60)
    logger.info("Starting Dependency Resolution")
    logger.info("=" * 60)
    
    if not state.settings.auto_install_deps:
        logger.info("Auto-install disabled in settings, skipping dependency resolution")
        return state
    
    all_imports = set()
    
    # Extract imports from source files
    logger.info(f"Analyzing {len(state.source_map)} source modules...")
    for module_path, code in state.source_map.items():
        logger.debug(f"  Scanning imports in {module_path}")
        imports = _extract_imports_from_code(code)
        all_imports.update(imports)
        logger.debug(f"    Found {len(imports)} imports")
    
    # Extract imports from generated test files
    logger.info(f"Analyzing {len(state.generated_tests)} test modules...")
    for module_path, test_code in state.generated_tests.items():
        logger.debug(f"  Scanning imports in test_{module_path}")
        imports = _extract_imports_from_code(test_code)
        all_imports.update(imports)
        logger.debug(f"    Found {len(imports)} imports")
    
    # Filter third-party packages using source_map
    third_party = _get_third_party_imports(all_imports, state)
    
    logger.info(f"Import Analysis Complete:")
    logger.info(f"  Total imports found: {len(all_imports)}")
    logger.info(f"  Standard library: {len(all_imports) - len(third_party) - len(_extract_local_module_names(state))}")
    logger.info(f"  Local modules: {len(_extract_local_module_names(state))}")
    logger.info(f"  Third-party packages: {len(third_party)}")
    
    if third_party:
        logger.info(f"Third-party packages detected: {sorted(third_party)}")
        
        # Install packages
        result = _install_packages(
            sorted(third_party),
            timeout=state.settings.pip_timeout
        )
        
        # Update state
        state.detected_dependencies = sorted(third_party)
        state.installed_dependencies = result.get("installed", [])
        state.failed_dependencies = result.get("failed", [])
        
        # Store in execution report for logging
        if "dependency_resolution" not in state.execution_report:
            state.execution_report["dependency_resolution"] = {}
        state.execution_report["dependency_resolution"] = result
        
        # Log summary
        if result.get("already_installed"):
            logger.info(f"Already installed: {len(result['already_installed'])} packages")
        if result.get("installed"):
            logger.info(f"✓ Newly installed: {len(result['installed'])} packages")
        if result.get("failed"):
            logger.warning(f"✗ Failed to install: {len(result['failed'])} packages")
            logger.warning(f"  Failed packages: {result['failed']}")
            logger.warning("  Tests may fail due to missing dependencies")
    else:
        logger.info("✓ No third-party dependencies detected")
        state.execution_report["dependency_resolution"] = {
            "status": "skipped",
            "message": "No third-party packages found"
        }
    
    logger.info("=" * 60)
    logger.info("Dependency Resolution Complete")
    logger.info("=" * 60)
    
    return state
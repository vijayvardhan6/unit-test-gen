from __future__ import annotations
import logging
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from typing import Dict

from schema import AgentState

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def _parse_junit(xml_path: str) -> Dict[str, int]:
    """Parse a pytest-generated JUnit XML to extract totals."""
    if not os.path.exists(xml_path):
        logger.warning(f"JUnit XML not found: {xml_path}")
        return {}
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        totals = dict(tests=0, errors=0, failures=0, skipped=0)
        suites = root.findall(".//testsuite")
        if not suites and root.tag == "testsuite":
            suites = [root]
        for s in suites:
            totals["tests"] += int(s.attrib.get("tests", 0))
            totals["errors"] += int(s.attrib.get("errors", 0))
            totals["failures"] += int(s.attrib.get("failures", 0))
            totals["skipped"] += int(s.attrib.get("skipped", 0))
        return totals
    except Exception as e:
        logger.error(f"Failed to parse JUnit XML: {e}")
        return {}

def _build_pythonpath(state: AgentState) -> str:
    """Build PYTHONPATH from project root and source directories."""
    python_paths = []
    
    # Add project root first
    if state.settings.project_root:
        project_root = os.path.abspath(state.settings.project_root)
        if os.path.exists(project_root):
            python_paths.append(project_root)
            logger.debug(f"Added project root to PYTHONPATH: {project_root}")
    
    # Add source directories
    for path in state.paths:
        if os.path.isfile(path):
            src_dir = os.path.dirname(os.path.abspath(path))
        elif os.path.isdir(path):
            src_dir = os.path.abspath(path)
        else:
            continue
        
        if src_dir not in python_paths and os.path.exists(src_dir):
            python_paths.append(src_dir)
            logger.debug(f"Added source directory to PYTHONPATH: {src_dir}")
    
    return os.pathsep.join(python_paths)

def executor_node(state: AgentState) -> AgentState:
    """
    Run pytest via subprocess with proper environment configuration.
    Uses subprocess instead of pytest.main() for better isolation.
    """
    if state.iteration_count > 0:
        logger.info("Skipping dependency resolution (already installed)")
        return state

    logger.info("Starting test execution")
    
    if not state.test_file:
        logger.error("No test file specified in state")
        state.execution_report = {"error": "No test file specified"}
        return state
    
    if not os.path.exists(state.test_file):
        logger.error(f"Test file does not exist: {state.test_file}")
        state.execution_report = {"error": f"Test file not found: {state.test_file}"}
        return state

    try:
        # Prepare reports directory
        reports_dir = os.path.join(os.getcwd(), "reports", "executor")
        os.makedirs(reports_dir, exist_ok=True)
        junit_xml = os.path.join(reports_dir, "junit.xml")
        logger.debug(f"Reports directory: {reports_dir}")
        
        # Build environment with PYTHONPATH
        env = os.environ.copy()
        pythonpath = _build_pythonpath(state)
        if pythonpath:
            existing_path = env.get('PYTHONPATH', '')
            env['PYTHONPATH'] = f"{pythonpath}{os.pathsep}{existing_path}" if existing_path else pythonpath
            logger.info(f"PYTHONPATH: {env['PYTHONPATH']}")

        test_dir = os.path.join(os.getcwd(), state.settings.test_root)
        
        # Build pytest command
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            test_dir,
            f"--junitxml={junit_xml}",
            "-v",  # Verbose output
        ] + list(state.settings.pytest_args or [])
        
        logger.info(f"Executing: {' '.join(cmd)}")
        logger.info(f"Test file: {state.test_file}")
        logger.info(f"Working directory: {os.getcwd()}")
        
        # Run pytest as subprocess
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            cwd=os.getcwd(),
            timeout=1800  # 30 minute timeout
        )
        
        logger.info(f"Pytest completed with return code: {result.returncode}")
        
        # Analyze exit code
        if result.returncode == 0:
            logger.info("All tests passed")
        elif result.returncode == 1:
            logger.warning("Some tests failed")
        elif result.returncode == 2:
            logger.error("Test execution was interrupted")
        elif result.returncode == 3:
            logger.error("Internal pytest error")
        elif result.returncode == 4:
            logger.error("pytest command line usage error")
        elif result.returncode == 5:
            logger.error("No tests collected")
        
        # Log output for debugging
        if result.stdout:
            logger.debug(f"Stdout:\n{result.stdout}")
        if result.stderr:
            logger.warning(f"Stderr:\n{result.stderr}")
        
        # Detect common issues
        stderr_lower = result.stderr.lower()
        if "modulenotfounderror" in stderr_lower or "importerror" in stderr_lower:
            logger.error("⚠️  Import errors detected - check module paths and PYTHONPATH")
        elif "syntaxerror" in stderr_lower:
            logger.error("⚠️  Syntax errors in test code")
        elif "collection errors" in stderr_lower:
            logger.error("⚠️  Test collection failed")
        
        # Parse JUnit XML for detailed results
        totals = _parse_junit(junit_xml)
        if totals:
            logger.info(f"Test summary: {totals['tests']} tests, "
                       f"{totals['failures']} failures, "
                       f"{totals['errors']} errors, "
                       f"{totals['skipped']} skipped")
        else:
            logger.warning("No test summary available from JUnit XML")
        
        # Update state
        state.junit_xml_path = junit_xml
        state.execution_report = {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "summary": totals,
            "success": result.returncode == 0,
        }
        
        return state
        
    except subprocess.TimeoutExpired:
        logger.error("Test execution timed out after 5 minutes")
        state.execution_report = {"error": "Test execution timeout"}
        return state
    except FileNotFoundError as e:
        logger.error(f"pytest not found: {e}")
        state.execution_report = {"error": "pytest not installed or not in PATH"}
        return state
    except Exception as e:
        logger.error(f"Test execution failed: {type(e).__name__}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        state.execution_report = {"error": f"{type(e).__name__}: {e}"}
        return state
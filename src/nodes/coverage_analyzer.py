from __future__ import annotations
import json
import logging
import os
from typing import Dict, Any, List

from schema import AgentState

# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def _read_coverage_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        logger.warning(f"Coverage JSON file does not exist: {path}")
        return {}
    try:
        logger.debug(f"Reading coverage JSON from: {path}")
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
            logger.debug(f"Successfully loaded coverage JSON with keys: {list(data.keys())}")
            return data
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse coverage JSON: {e}")
        return {}
    except Exception as e:
        logger.error(f"Error reading coverage JSON: {e}")
        return {}

def _summarize_coverage_json(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not payload:
        return {"summary": {}, "per_file": {}}
    totals = payload.get("totals", {})
    files = payload.get("files", {})
    per_file = {}
    for fpath, info in files.items():
        summ = info.get("summary", {})
        per_file[fpath] = {
            "percent": summ.get("percent_covered"),
            "covered": summ.get("covered_lines"),
            "statements": summ.get("num_statements"),
            "missing": (summ.get("num_statements") or 0) - (summ.get("covered_lines") or 0),
        }
    return {
        "summary": {
            "percent": totals.get("percent_covered"),
            "covered": totals.get("covered_lines"),
            "statements": totals.get("num_statements"),
            "missing": (totals.get("num_statements") or 0) - (totals.get("covered_lines") or 0),
        },
        "per_file": per_file,
    }

def coverage_analyzer_node(state: AgentState) -> AgentState:
    """
    Run pytest with coverage using pytest-cov plugin for accurate measurement.
    """
    logger.info("Starting coverage analysis")

    try:
        import subprocess
        import sys  
        import time

        reports_dir = os.path.join(os.getcwd(), "reports", "coverage")
        os.makedirs(reports_dir, exist_ok=True) 
        cov_json = os.path.join(reports_dir, "coverage.json")
        cov_xml = os.path.join(reports_dir, "coverage.xml")
        html_dir = os.path.join(reports_dir, "html")
        logger.debug(f"Created reports directory: {reports_dir}")

        test_dir = os.path.join(os.getcwd(), state.settings.test_root)
        # test_dir = os.path.join(os.getcwd(), "tests")

        # Build pytest command with coverage
        # USE sys.executable instead of "python"
        cmd = [
            sys.executable,  # CHANGED: Use the current Python interpreter
            "-m", 
            "pytest",
            "-q",
            test_dir,
            "--cov=src/data",
            f"--cov-report=json:{cov_json}",
            f"--cov-report=xml:{cov_xml}",
            f"--cov-report=html:{html_dir}",
            "--cov-report=term-missing",
            "--cov-branch",
        ]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        logger.info(f"Using Python executable: {sys.executable}")  # Log which Python is being used
        
        # Run pytest with coverage as a subprocess
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=os.getcwd()
        )
        
        logger.info(f"Pytest completed with return code: {result.returncode}")
        
        # Log the actual output to see what happened
        if result.stdout:
            logger.info(f"Pytest stdout:\n{result.stdout}")
        if result.stderr:
            logger.warning(f"Pytest stderr:\n{result.stderr}")
        
        # Give the file system a moment to finish writing files
        time.sleep(0.5)
        
        # Check if coverage.json was created
        if not os.path.exists(cov_json):
            logger.error(f"Coverage JSON file not found at: {cov_json}")
            if os.path.exists(reports_dir):
                logger.info(f"Files in reports directory: {os.listdir(reports_dir)}")
            state.coverage_report = {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "error": "Coverage JSON file not generated",
            }
            return state
        
        logger.info(f"Coverage JSON file found at: {cov_json}")
        logger.info(f"File size: {os.path.getsize(cov_json)} bytes")
        
        # Read and parse coverage results
        payload = _read_coverage_json(cov_json)
        
        if not payload:
            logger.error("Failed to read coverage JSON payload")
            try:
                with open(cov_json, 'r') as f:
                    content = f.read()
                    logger.debug(f"Coverage file content (first 500 chars):\n{content[:500]}")
            except Exception as e:
                logger.error(f"Could not read coverage file: {e}")
        
        summary = _summarize_coverage_json(payload)
        
        if summary and summary["summary"]:
            coverage_stats = summary["summary"]
            logger.info(f"Coverage summary: {coverage_stats['percent']:.1f}% covered, "
                       f"{coverage_stats['covered']} of {coverage_stats['statements']} statements, "
                       f"{coverage_stats['missing']} missing")
            
            # Log per-file coverage with detailed breakdown
            for fpath, stats in summary.get("per_file", {}).items():
                logger.info(f"  {fpath}: {stats['percent']:.1f}% "
                           f"({stats['covered']}/{stats['statements']} statements, "
                           f"{stats['missing']} missing)")
        else:
            logger.warning("No coverage summary available")
        
        logger.info(f"Coverage HTML report available at: {html_dir}/index.html")

        state.coverage_json_path = cov_json
        state.coverage_report = {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "summary": summary,
            "artifacts": {"json": cov_json, "xml": cov_xml, "html": html_dir},
        }
        return state
    except ImportError as e:
        logger.error(f"pytest or pytest-cov not available: {e.__class__.__name__}")
        state.coverage_report = {
            "skipped": True,
            "reason": f"pytest/pytest-cov not importable: {e.__class__.__name__}",
        }
        return state
    except Exception as e:
        logger.error(f"Coverage analysis failed: {type(e).__name__}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        state.coverage_report = {
            "error": f"{type(e).__name__}: {e}",
        }

        current_coverage = summary.get('summary', {}).get('percent', 0)
        if state.iteration_count > 0:
            state.previous_coverage = current_coverage
            
        return state
import logging
import os
import re
from typing import Tuple
from schema import AgentState

# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def _target_test_file_for(module_path: str, test_root: str) -> Tuple[str, str]:
    # Normalize file name
    base = module_path.replace(os.sep, "_").replace(".", "_").replace(":", "_")
    if base.endswith("_py"):
        base = base[:-3]
    fname = f"test_{base}.py"
    out_dir = os.path.join(os.getcwd(), test_root)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir, os.path.join(out_dir, fname)

def write_node(state: AgentState) -> AgentState:
    logger.info("Starting to write generated tests")
    
    if not state.generated_tests:
        logger.info("No generated tests to write")
        return state
        
    logger.info(f"Writing tests for {len(state.generated_tests)} modules")
    
    for module_path, test_code in state.generated_tests.items():
        logger.info(f"Processing module: {module_path}")
        out_dir, outfile = _target_test_file_for(module_path, state.settings.test_root)
        logger.debug(f"Target directory: {out_dir}")
        logger.debug(f"Target file: {outfile}")
        
        content = test_code.strip() + "\n"
        
        try:
            # Append if file exists
            if os.path.exists(outfile):
                logger.debug(f"Appending to existing file: {outfile}")
                with open(outfile, "a", encoding="utf-8") as fh:
                    fh.write("\n\n# ---- Appended block ----\n")
                    fh.write(content)
                logger.info(f"Successfully appended tests to {outfile}")
            else:
                logger.debug(f"Creating new file: {outfile}")
                with open(outfile, "w", encoding="utf-8") as fh:
                    fh.write(content)
                logger.info(f"Successfully created new test file: {outfile}")
            
            state.test_file = outfile
            logger.info(f"Test file: {state.test_file}")
        except Exception as e:
            logger.error(f"Failed to write tests for {module_path}: {str(e)}")
            continue
    
    logger.info("Completed writing all test files")
    return state

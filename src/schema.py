from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any


class Settings(BaseModel):
    model: str = "openai/gpt-oss-20b"
    temperature: float = 0.2
    max_output_tokens: int = 2000
    project_root: str = "" 
    test_root: str = "tests"
    exclude_dirs: List[str] = Field(default_factory=lambda: ["tests", ".venv", "venv", "__pycache__"])
    enable_coverage: bool = True
    pytest_args: List[str] = Field(default_factory=lambda: ["-q"])
    auto_install_deps: bool = True
    pip_timeout: int = 600
    
    enable_pattern_classification: bool = True
    enable_smart_prioritization: bool = True
    max_prioritized_targets: int = 10  # Top N targets to focus on per iteration


class SourceTarget(BaseModel):
    module_path: str
    symbol: str
    qualified_name: str
    target_type: str = "function"  # "function", "class", "method"
    
    # OOP-specific fields
    parent_class: Optional[str] = None
    class_methods: List[str] = Field(default_factory=list)
    class_init_signature: Optional[str] = None
    class_attributes: List[str] = Field(default_factory=list)
    is_dataclass: bool = False
    base_classes: List[str] = Field(default_factory=list)


class PlanItem(BaseModel):
    name: str
    brief_summary: str
    behaviors: List[str]
    example_inputs: List[str]



class CoveragePattern(BaseModel):
    """Classification of coverage gap patterns."""
    pattern_type: str  # "branch", "exception", "edge_case", "loop", "state_mutation", "integration"
    target_name: str
    module_path: str
    uncovered_lines: List[int]
    confidence: float  # 0-1, how confident we are about this pattern
    priority: int  # 1-5, urgency (5 = highest)
    suggested_test_approach: str  # Hint for test generation
    complexity_score: float  # Estimated difficulty
    context: str  # Code snippet showing the pattern


class PrioritizedTarget(BaseModel):
    """Target with prioritization metadata."""
    target_name: str
    module_path: str
    priority_score: float  # Combined score for ranking
    patterns: List[CoveragePattern]
    current_coverage: float
    potential_gain: float
    reason: str  # Why this was prioritized




class AgentState(BaseModel):
    user_input: str = ""
    paths: List[str] = Field(default_factory=list)
    inline_code: Optional[str] = None
    source_map: Dict[str, str] = Field(default_factory=dict)
    source_files: List[str] = Field(default_factory=list)
    targets: List[SourceTarget] = Field(default_factory=list)
    plan: List[PlanItem] = Field(default_factory=list)
    generated_tests: Dict[str, str] = Field(default_factory=dict)
    review_notes: Dict[str, str] = Field(default_factory=dict)
    settings: Settings = Field(default_factory=Settings)
    test_file: Optional[str] = None
    
    target_contexts: Dict[str, str] = Field(default_factory=dict)
    
    # Dependencies
    detected_dependencies: List[str] = Field(default_factory=list)
    installed_dependencies: List[str] = Field(default_factory=list)
    failed_dependencies: List[str] = Field(default_factory=list)
    
    # Execution and coverage
    execution_report: Dict[str, Any] = Field(default_factory=dict)
    coverage_report: Dict[str, Any] = Field(default_factory=dict)
    junit_xml_path: Optional[str] = None
    coverage_json_path: Optional[str] = None
    
    # Feedback loop
    iteration_count: int = 0
    max_iterations: int = 1
    coverage_threshold: float = 80.0
    previous_coverage: Optional[float] = None
    failing_functions: List[Dict[str, Any]] = Field(default_factory=list)
    supplemental_tests: Dict[str, List[str]] = Field(default_factory=dict)
    final_report: Dict[str, Any] = Field(default_factory=dict)
    
    # Pattern classification
    coverage_patterns: List[CoveragePattern] = Field(default_factory=list)
    patterns_by_target: Dict[str, List[CoveragePattern]] = Field(default_factory=dict)
    
    # Smart prioritization
    prioritized_targets: List[PrioritizedTarget] = Field(default_factory=list)
    skipped_targets: List[str] = Field(default_factory=list)  # Low ROI targets
    
    # Learning data (for future ML)
    iteration_improvements: List[float] = Field(default_factory=list)  # Coverage gain per iteration
    pattern_accuracy: List[Dict[str, Any]] = Field(default_factory=list)  # Track prediction accuracy
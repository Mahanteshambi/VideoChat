"""Tests for annotator modules."""

import pytest
from pathlib import Path
from videoindex.annotators import ShotInfo, AnnotationResult


def test_shot_info():
    """Test ShotInfo dataclass."""
    shot = ShotInfo(
        shot_number=1,
        start_time_seconds=0.0,
        end_time_seconds=5.0,
        duration_seconds=5.0
    )
    
    assert shot.shot_number == 1
    assert shot.start_time_seconds == 0.0
    assert shot.end_time_seconds == 5.0
    assert shot.duration_seconds == 5.0


def test_annotation_result():
    """Test AnnotationResult dataclass."""
    shot_info = ShotInfo(
        shot_number=1,
        start_time_seconds=0.0,
        end_time_seconds=5.0,
        duration_seconds=5.0
    )
    
    result = AnnotationResult(
        shot_info=shot_info,
        metadata={"description": "test shot"},
        model_name="TestModel",
        processing_time_seconds=1.5,
        success=True
    )
    
    assert result.shot_info.shot_number == 1
    assert result.metadata["description"] == "test shot"
    assert result.model_name == "TestModel"
    assert result.processing_time_seconds == 1.5
    assert result.success is True
    
    # Test to_dict method
    result_dict = result.to_dict()
    assert result_dict["shot_info"]["shot_number"] == 1
    assert result_dict["metadata"]["description"] == "test shot"
    assert result_dict["model_name"] == "TestModel"
    assert result_dict["success"] is True

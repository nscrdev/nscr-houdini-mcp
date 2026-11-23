import json

import pytest
from mcp_types import CallToolResult, TextContent

import sequence


def job_step(**extra) -> sequence.Step:
    body = {"job_id": "job-shape", "state": "done", "outputs": {"result": 3}, **extra}
    result = CallToolResult(
        content=[TextContent(type="text", text=json.dumps(body))], structured_content=body
    )
    return sequence.Step(
        label="python build, followed",
        tool="hou_jobs",
        arguments={},
        elapsed_s=0.1,
        result=result,
        is_error=False,
        structured=True,
        text="mirror",
        image=False,
        blocks=["text"],
    )


def test_a_jobs_export_timing_does_not_change_its_protocol_shape() -> None:
    before = job_step()
    after = job_step(export_path="/project/.agent/jobs/job-shape.json")
    assert sequence.shape(before) == sequence.shape(after)
    assert "export_path" in after.body


def test_an_export_path_with_the_wrong_type_is_not_hidden() -> None:
    with pytest.raises(sequence.SequenceFailed):
        sequence.shape(job_step(export_path=123))


def test_other_job_shape_differences_are_still_seen() -> None:
    assert sequence.shape(job_step()) != sequence.shape(job_step(unexpected=True))

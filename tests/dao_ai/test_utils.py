import pytest

from dao_ai.utils import (
    _extract_first_json_object,
    _repair_json,
    is_lib_provided,
    load_function,
)


@pytest.mark.unit
def test_load_function_valid_builtin() -> None:
    """Test loading a valid built-in function."""
    func = load_function("builtins.len")
    assert callable(func)
    assert func([1, 2, 3]) == 3


@pytest.mark.unit
def test_load_function_valid_module_function() -> None:
    """Test loading a valid function from a standard library module."""
    func = load_function("os.path.join")
    assert callable(func)
    assert func("path", "to", "file") == "path/to/file"


@pytest.mark.unit
def test_load_function_invalid_module() -> None:
    """Test loading a function from a non-existent module."""
    with pytest.raises(ImportError, match="Failed to import nonexistent.module.func"):
        load_function("nonexistent.module.func")


@pytest.mark.unit
def test_load_function_invalid_function() -> None:
    """Test loading a non-existent function from a valid module."""
    with pytest.raises(ImportError, match="Failed to import os.nonexistent_function"):
        load_function("os.nonexistent_function")


@pytest.mark.unit
def test_load_function_non_callable() -> None:
    """Test loading a non-callable attribute."""
    with pytest.raises(ImportError, match="Failed to import os.name"):
        load_function("os.name")


@pytest.mark.unit
def test_load_function_langchain_tool() -> None:
    """Test loading a langchain tool (StructuredTool).

    In langchain 1.x, tools decorated with @tool return StructuredTool objects
    which are not directly callable (callable() returns False) but have an
    invoke() method.
    """
    tool = load_function("dao_ai.tools.current_time_tool")

    # Verify it's a langchain tool
    assert hasattr(tool, "invoke")
    assert hasattr(tool, "name")
    assert tool.name == "current_time_tool"

    # Verify it can be invoked
    result = tool.invoke({})
    assert isinstance(result, str)


@pytest.mark.unit
def test_load_function_no_dot_separator() -> None:
    """Test loading with invalid function name format."""
    with pytest.raises(ValueError):
        load_function("invalid_format")


@pytest.mark.unit
def test_is_lib_provided() -> None:
    """Test if a library is provided in the pip requirements."""
    assert is_lib_provided("dao-ai", ["dao-ai", "pandas"]) is True
    assert is_lib_provided("dao-ai", ["dao-ai>=0.0.1", "pandas"]) is True
    assert is_lib_provided("dao-ai", ["numpy", "pandas"]) is False
    assert (
        is_lib_provided(
            "dao-ai", ["git+https://github.com/natefleming/dao-ai.git", "numpy"]
        )
        is True
    )


@pytest.mark.unit
class TestExtractFirstJsonObject:
    """Tests for extracting the first complete JSON object from mixed text."""

    def test_pure_json(self) -> None:
        """Extract from clean JSON."""
        content = '{"queries": [{"text": "test"}]}'
        result = _extract_first_json_object(content)
        assert result == content

    def test_json_with_text_before(self) -> None:
        """Extract JSON with leading text."""
        content = 'Some text before {"key": "value"}'
        result = _extract_first_json_object(content)
        assert result == '{"key": "value"}'

    def test_json_with_text_after(self) -> None:
        """Extract JSON with trailing text."""
        content = '{"key": "value"} and some text after'
        result = _extract_first_json_object(content)
        assert result == '{"key": "value"}'

    def test_multiple_json_objects(self) -> None:
        """Extract first JSON when multiple objects present (small model behavior)."""
        content = (
            "I'll help you find Milwaukee drills. "
            '{"queries": [{"text": "Milwaukee drills", "filters": []}]}'
            "Let me try a broader search: "
            '{"queries": [{"text": "drill", "filters": []}]}'
        )
        result = _extract_first_json_object(content)
        assert result == '{"queries": [{"text": "Milwaukee drills", "filters": []}]}'

    def test_nested_objects(self) -> None:
        """Handle nested JSON objects correctly."""
        content = '{"outer": {"inner": {"deep": "value"}}}'
        result = _extract_first_json_object(content)
        assert result == content

    def test_string_with_braces(self) -> None:
        """Handle strings containing brace characters."""
        content = '{"text": "value with { and } inside"}'
        result = _extract_first_json_object(content)
        assert result == content

    def test_escaped_quotes(self) -> None:
        """Handle escaped quotes in strings."""
        content = '{"text": "value with \\"escaped\\" quotes"}'
        result = _extract_first_json_object(content)
        assert result == content

    def test_no_json(self) -> None:
        """Return None when no JSON present."""
        content = "Just some plain text"
        result = _extract_first_json_object(content)
        assert result is None


@pytest.mark.unit
class TestRepairJson:
    """Tests for JSON repair functionality."""

    def test_valid_json_unchanged(self) -> None:
        """Valid JSON passes through unchanged."""
        content = '{"key": "value"}'
        result = _repair_json(content)
        assert result == content

    def test_extracts_from_mixed_text(self) -> None:
        """Extract JSON from text with content before and after."""
        content = "Here is the JSON: {\"key\": \"value\"} That was it."
        result = _repair_json(content)
        assert result == '{"key": "value"}'

    def test_multiple_json_extracts_first(self) -> None:
        """When multiple JSON objects present, extract the first complete one."""
        content = (
            'First: {"a": 1} Second: {"b": 2}'
        )
        result = _repair_json(content)
        assert result == '{"a": 1}'

    def test_small_model_streaming_pattern(self) -> None:
        """Handle the small model pattern of streaming JSON mixed with text."""
        content = (
            "I'll help you find Milwaukee drills under $200. "
            '{"queries": [{"text": "Milwaukee drills", "filters": [{"key": "brand", "value": "MILWAUKEE"}]}]}'
            "Let me try another search: "
            '{"queries": [{"text": "drills", "filters": []}]}'
        )
        result = _repair_json(content)
        assert result is not None
        import json
        parsed = json.loads(result)
        assert parsed["queries"][0]["text"] == "Milwaukee drills"

    def test_fixes_trailing_comma(self) -> None:
        """Fix trailing comma before closing brace."""
        content = '{"key": "value",}'
        result = _repair_json(content)
        assert result == '{"key": "value"}'

    def test_fixes_trailing_comma_in_array(self) -> None:
        """Fix trailing comma before closing bracket."""
        content = '{"arr": [1, 2, 3,]}'
        result = _repair_json(content)
        assert result == '{"arr": [1, 2, 3]}'

    def test_closes_unclosed_braces(self) -> None:
        """Close unclosed braces in truncated JSON."""
        content = '{"key": {"nested": "value"'
        result = _repair_json(content)
        assert result == '{"key": {"nested": "value"}}'

    def test_closes_unclosed_brackets(self) -> None:
        """Close unclosed brackets in truncated JSON."""
        content = '{"arr": [1, 2, 3'
        result = _repair_json(content)
        assert result == '{"arr": [1, 2, 3]}'

    def test_no_json_returns_none(self) -> None:
        """Return None when no JSON present."""
        content = "No JSON here at all"
        result = _repair_json(content)
        assert result is None

    def test_severely_malformed_returns_none(self) -> None:
        """Return None for severely malformed content."""
        content = '{"key": broken without quotes}'
        result = _repair_json(content)
        assert result is None

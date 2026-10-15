"""Keep Function registration readable without importing or deploying modules."""
import ast
from pathlib import Path

import pytest


def violations(source):
    tree = ast.parse(source)
    modules = {"chalkcompute"}
    function_names = set()
    remote_names = {"RemoteFunction"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.asname or alias.name for alias in node.names if alias.name == "chalkcompute")
        elif isinstance(node, ast.ImportFrom) and node.module == "chalkcompute":
            for alias in node.names:
                if alias.name == "function":
                    function_names.add(alias.asname or alias.name)
                elif alias.name == "RemoteFunction":
                    remote_names.add(alias.asname or alias.name)

    # Only the callable in a literal @chalkcompute.function(...) decorator is
    # allowed. Even assigning it to a factory variable is a style regression.
    allowed = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for decorator in node.decorator_list:
                if (isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute)
                        and isinstance(decorator.func.value, ast.Name)
                        and decorator.func.value.id == "chalkcompute" and decorator.func.attr == "function"):
                    allowed.add(id(decorator.func))

    errors = []
    for node in ast.walk(tree):
        function_reference = (
            isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in function_names
        ) or (
            isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
            and node.value.id in modules and node.attr == "function"
        )
        if function_reference and id(node) not in allowed:
            errors.append(f"line {node.lineno}: use @chalkcompute.function(...) directly")
        if isinstance(node, ast.Call):
            target = node.func
            constructor = (
                isinstance(target, ast.Name) and target.id in remote_names
            ) or (
                isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)
                and target.value.id in modules and target.attr == "RemoteFunction"
            )
            if constructor or isinstance(target, ast.Attribute) and target.attr == "deploy":
                errors.append(f"line {node.lineno}: decoration owns Function deployment")
    return errors


def test_repository_only_registers_functions_with_clear_decorators():
    root = Path(__file__).resolve().parents[1]
    folders = ("semantic_ids", "src", "train", "host", "materialize", "train-recommender", "serve")
    files = [*root.glob("*.py"), *(path for folder in folders for path in (root / folder).rglob("*.py"))]
    errors = [f"{path.relative_to(root)}: {error}" for path in files
              for error in violations(path.read_text())]
    assert not errors, "\n".join(errors)


@pytest.mark.parametrize("source", [
    "import chalkcompute\nf = chalkcompute.function(image=image)(handler)",
    "import chalkcompute as cc\nfactory = cc.function\nf = factory()(handler)",
    "from chalkcompute import function as register\nf = register()(handler)",
    "from chalkcompute import RemoteFunction as RF\nf = RF(handler)",
    "import chalkcompute\nf = chalkcompute.RemoteFunction(handler)",
    "f.deploy()",
])
def test_rejects_alternative_registration_patterns(source):
    assert violations(source)


def test_allows_decorator_readiness_and_existing_function_invocation():
    source = """
import chalkcompute
from chalkcompute import RemoteFunction

@chalkcompute.function(image=image)
def predict(value: str) -> str:
    return value

predict.wait_ready()
existing = RemoteFunction.from_name("predict")
existing.remote("example")
"""
    assert violations(source) == []

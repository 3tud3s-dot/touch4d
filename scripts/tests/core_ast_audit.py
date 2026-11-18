"""Stable source-audit AST hashes across the supported Python versions.

Python 3.12 added empty ``type_params`` fields to function and class nodes.
Omitting that empty metadata is the sole cross-version normalization. Actual
type parameters are rejected rather than silently erased from the audit.
"""
import ast
import hashlib
import json


FORMAT = "touch4d-ast-json-v1-empty-type-params-only"


def canonical_ast(node):
    """Serialize every semantic AST field; source locations are not AST fields."""
    def encode(value):
        if isinstance(value, ast.AST):
            if getattr(value, "type_params", []) != []:
                raise ValueError("Nonempty AST type_params cannot be normalized")
            fields = []
            for name, item in ast.iter_fields(value):
                if name == "type_params":
                    continue
                fields.append([name, encode(item)])
            return [type(value).__name__, fields]
        if isinstance(value, list):
            return [encode(item) for item in value]
        if isinstance(value, bytes):
            return ["bytes", value.hex()]
        if isinstance(value, complex):
            return ["complex", value.real.hex(), value.imag.hex()]
        if isinstance(value, float):
            return ["float", value.hex()]
        if value is Ellipsis:
            return ["ellipsis"]
        if value is None or isinstance(value, (str, bool, int)):
            return value
        raise TypeError(f"Unsupported AST field type: {type(value).__name__}")
    return json.dumps([FORMAT, encode(node)], ensure_ascii=True, separators=(",", ":"))


def ast_sha256(node):
    return hashlib.sha256(canonical_ast(node).encode("utf-8")).hexdigest()

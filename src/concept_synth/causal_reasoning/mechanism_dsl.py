"""Restricted S-expression DSL utilities for CIND mechanism formulas."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Set, Tuple

_ALLOWED_SYMBOL = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_TOKEN_RE = re.compile(r"\(|\)|[^\s()]+")

CONST_TOKENS = {"0", "1"}


class MechanismParseError(ValueError):
    """Raised when a mechanism string cannot be parsed."""


class MechanismEvalError(ValueError):
    """Raised when a parsed mechanism cannot be evaluated."""


OPERATOR_ARITY = {
    "not": (1, 1),
    "and": (2, None),
    "or": (2, None),
    "xor": (2, None),
    "iff": (2, None),
    "if": (3, 3),
}

DEFAULT_ALLOWED_OPERATORS = ("not", "and", "or", "xor", "iff")


@dataclass(frozen=True)
class MechanismNode:
    """AST node for the restricted mechanism DSL."""

    kind: str  # "var", "const", or "op"
    value: str
    args: Tuple["MechanismNode", ...] = ()


class _TokenStream:
    def __init__(self, tokens: list[str]):
        self.tokens = tokens
        self.i = 0

    def has_more(self) -> bool:
        return self.i < len(self.tokens)

    def peek(self) -> Optional[str]:
        if self.i < len(self.tokens):
            return self.tokens[self.i]
        return None

    def pop(self) -> str:
        if self.i >= len(self.tokens):
            raise MechanismParseError("Unexpected end of expression")
        tok = self.tokens[self.i]
        self.i += 1
        return tok


def _tokenize(expr: str) -> list[str]:
    return _TOKEN_RE.findall(expr.strip())


def _is_symbol(token: str) -> bool:
    return bool(_ALLOWED_SYMBOL.match(token))


def _normalize_allowed_ops(
    allowed_operators: Optional[Tuple[str, ...] | list[str] | set[str]],
) -> Set[str]:
    if allowed_operators is None:
        return set(DEFAULT_ALLOWED_OPERATORS)
    return {op.strip() for op in allowed_operators if op and op.strip()}


def _normalize_allowed_variables(
    allowed_variables: Optional[Tuple[str, ...] | list[str] | set[str]],
) -> Optional[Set[str]]:
    if allowed_variables is None:
        return None

    out: Set[str] = set()
    for v in allowed_variables:
        s = str(v).strip()
        if not s:
            continue
        if not _is_symbol(s):
            raise MechanismParseError(f"Invalid allowed variable token: {s!r}")
        out.add(s)
    return out


def _check_arity(op: str, args: list[MechanismNode]) -> None:
    min_arity, max_arity = OPERATOR_ARITY[op]
    n = len(args)
    if n < min_arity:
        raise MechanismParseError(f"Operator '{op}' expects at least {min_arity} args, got {n}")
    if max_arity is not None and n > max_arity:
        raise MechanismParseError(f"Operator '{op}' expects at most {max_arity} args, got {n}")


def _parse_atom(token: str, allowed_vars: Optional[Set[str]]) -> MechanismNode:
    if token in CONST_TOKENS:
        return MechanismNode(kind="const", value=token)

    if not _is_symbol(token):
        raise MechanismParseError(f"Invalid symbol '{token}'")

    if allowed_vars is not None and token not in allowed_vars:
        raise MechanismParseError(f"Variable '{token}' is not allowed")

    return MechanismNode(kind="var", value=token)


def _parse_expr(
    ts: _TokenStream,
    allowed_ops: Set[str],
    allowed_vars: Optional[Set[str]],
    *,
    allow_constants: bool,
) -> MechanismNode:
    tok = ts.pop()

    if tok == "(":
        head = ts.pop()
        if head == ")":
            raise MechanismParseError("Empty list is not a valid mechanism expression")

        args: list[MechanismNode] = []
        while True:
            nxt = ts.peek()
            if nxt is None:
                raise MechanismParseError("Missing closing ')' in mechanism expression")
            if nxt == ")":
                ts.pop()
                break
            args.append(
                _parse_expr(
                    ts,
                    allowed_ops,
                    allowed_vars,
                    allow_constants=allow_constants,
                )
            )

        if not args:
            if head in CONST_TOKENS and not allow_constants:
                raise MechanismParseError("Constants 0/1 are not allowed")
            return _parse_atom(head, allowed_vars)

        if head not in OPERATOR_ARITY:
            raise MechanismParseError(f"Unknown operator '{head}'")
        if head not in allowed_ops:
            raise MechanismParseError(f"Operator '{head}' is not allowed")

        _check_arity(head, args)
        return MechanismNode(kind="op", value=head, args=tuple(args))

    if tok == ")":
        raise MechanismParseError("Unexpected ')' in mechanism expression")

    if tok in CONST_TOKENS and not allow_constants:
        raise MechanismParseError("Constants 0/1 are not allowed")
    return _parse_atom(tok, allowed_vars)


def parse_mechanism(
    expression: str,
    allowed_operators: Optional[Tuple[str, ...] | list[str] | set[str]] = None,
    allowed_variables: Optional[Tuple[str, ...] | list[str] | set[str]] = None,
    allow_constants: bool = True,
) -> MechanismNode:
    """Parse a restricted mechanism S-expression into an AST."""
    if not expression or not expression.strip():
        raise MechanismParseError("Empty mechanism expression")

    tokens = _tokenize(expression)
    if not tokens:
        raise MechanismParseError("Empty mechanism expression")

    allowed_ops = _normalize_allowed_ops(allowed_operators)
    allowed_vars = _normalize_allowed_variables(allowed_variables)

    ts = _TokenStream(tokens)
    node = _parse_expr(ts, allowed_ops, allowed_vars, allow_constants=bool(allow_constants))

    if ts.has_more():
        raise MechanismParseError("Trailing tokens after complete mechanism expression")

    return node


def _coerce_bool(value: Any, symbol: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value in (0, 1):
            return bool(value)
        raise MechanismEvalError(f"Variable '{symbol}' must be binary (0/1), got {value}")
    if isinstance(value, str):
        lower = value.strip().lower()
        if lower in {"0", "false"}:
            return False
        if lower in {"1", "true"}:
            return True
    raise MechanismEvalError(f"Variable '{symbol}' must be binary (0/1 or bool), got {value!r}")


def _eval(node: MechanismNode, assignment: Dict[str, Any]) -> bool:
    if node.kind == "const":
        return node.value == "1"

    if node.kind == "var":
        if node.value not in assignment:
            raise MechanismEvalError(f"Missing variable '{node.value}' in assignment")
        return _coerce_bool(assignment[node.value], node.value)

    op = node.value
    values = [_eval(arg, assignment) for arg in node.args]

    if op == "not":
        return not values[0]
    if op == "and":
        return all(values)
    if op == "or":
        return any(values)
    if op == "xor":
        return (sum(1 for v in values if v) % 2) == 1
    if op == "iff":
        head = values[0]
        return all(v == head for v in values[1:])
    if op == "if":
        cond, left, right = values
        return left if cond else right

    raise MechanismEvalError(f"Unsupported operator '{op}'")


def evaluate_parsed_mechanism(node: MechanismNode, assignment: Dict[str, Any]) -> int:
    """Evaluate a parsed mechanism node to an integer bit (0/1)."""
    return 1 if _eval(node, assignment) else 0


def evaluate_mechanism(
    expression: str,
    assignment: Dict[str, Any],
    allowed_operators: Optional[Tuple[str, ...] | list[str] | set[str]] = None,
    allowed_variables: Optional[Tuple[str, ...] | list[str] | set[str]] = None,
    allow_constants: bool = True,
) -> int:
    """Parse and evaluate a mechanism expression to 0/1."""
    node = parse_mechanism(
        expression,
        allowed_operators=allowed_operators,
        allowed_variables=allowed_variables,
        allow_constants=allow_constants,
    )
    return evaluate_parsed_mechanism(node, assignment)


def ast_size(node: MechanismNode) -> int:
    """Return AST node count."""
    if node.kind in {"var", "const"}:
        return 1
    return 1 + sum(ast_size(arg) for arg in node.args)


def ast_depth(node: MechanismNode) -> int:
    """Return maximum AST depth (root depth=1)."""
    if node.kind in {"var", "const"}:
        return 1
    return 1 + max(ast_depth(arg) for arg in node.args)


def _collect_operator_counts(node: MechanismNode, counts: Dict[str, int]) -> None:
    if node.kind == "op":
        counts[node.value] = counts.get(node.value, 0) + 1
        for arg in node.args:
            _collect_operator_counts(arg, counts)


def _collect_variables(node: MechanismNode, out: Set[str]) -> None:
    if node.kind == "var":
        out.add(node.value)
        return
    for arg in node.args:
        _collect_variables(arg, out)


def mechanism_variables(node: MechanismNode) -> Set[str]:
    out: Set[str] = set()
    _collect_variables(node, out)
    return out


def analyze_mechanism(
    expression_or_node: str | MechanismNode,
    allowed_operators: Optional[Tuple[str, ...] | list[str] | set[str]] = None,
    allowed_variables: Optional[Tuple[str, ...] | list[str] | set[str]] = None,
    allow_constants: bool = True,
) -> Dict[str, Any]:
    """Compute complexity metadata for a mechanism expression."""
    if isinstance(expression_or_node, MechanismNode):
        node = expression_or_node
    else:
        node = parse_mechanism(
            expression_or_node,
            allowed_operators=allowed_operators,
            allowed_variables=allowed_variables,
            allow_constants=allow_constants,
        )

    op_counts: Dict[str, int] = {}
    _collect_operator_counts(node, op_counts)

    vars_used = sorted(mechanism_variables(node))
    return {
        "astSize": ast_size(node),
        "maxDepth": ast_depth(node),
        "operatorCounts": op_counts,
        "variables": vars_used,
        "parentCount": len(vars_used),
        "containsIf": op_counts.get("if", 0) > 0,
    }


def node_to_sexpr(node: MechanismNode) -> str:
    """Serialize a parsed node back to S-expression."""
    if node.kind in {"var", "const"}:
        return node.value
    return "(" + " ".join([node.value] + [node_to_sexpr(arg) for arg in node.args]) + ")"

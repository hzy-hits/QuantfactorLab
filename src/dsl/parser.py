"""
Pratt parser for the Factor Lab DSL.

Supports:
  - Function calls:   ts_mean(close, 20), rank(x), if_then(a, b, c)
  - Infix operators:  + - * / with correct precedence
  - Unary minus:      -ret_5d
  - Nested exprs:     rank(delta(ts_mean(close, 5), 10))
  - Features (idents): close, volume, ret_5d, ...
  - Numeric literals:  5, 20, 0.5, 1e-9

Grammar (informally):
  expr     = unary ((+|-|*|/) unary)*       # Pratt handles precedence
  unary    = '-' unary | primary
  primary  = NUMBER | IDENT '(' args ')' | IDENT | '(' expr ')'
  args     = expr (',' expr)*
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List


# ---------------------------------------------------------------------------
# Safety constants
# ---------------------------------------------------------------------------

MAX_EXPRESSION_LENGTH = 200
MAX_AST_DEPTH = 4

ALLOWED_LOOKBACK_WINDOWS: set[int] = {1, 2, 3, 5, 10, 14, 20, 30, 40, 60, 120, 250}

ALLOWED_FUNCTIONS: set[str] = {
    # time-series
    "ts_mean", "ts_std", "ts_max", "ts_min", "ts_sum", "ts_rank",
    "ts_argmax", "ts_argmin", "ts_corr", "ts_cov",
    "ts_count", "ts_skew", "ts_kurt", "ts_product",
    "delta", "pct_change", "shift", "decay_linear", "decay_exp",
    # cross-sectional
    "rank", "zscore", "demean",  # quantile/neutralize not yet in operators.py
    # universal
    "abs", "sign", "log", "sqrt", "power", "clamp", "max", "min", "if_then",
}

FEATURE_WHITELIST: set[str] = {
    "open", "high", "low", "close", "vwap",
    "volume", "amount", "turnover_rate", "volume_ratio",
    "ret_1d", "ret_5d", "ret_20d", "ret_60d",
    "rsi_14", "bb_position", "ma20_dist", "ma60_dist", "atr_14", "obv",
    "pe_ttm", "pb", "market_cap", "circ_market_cap", "ps_ttm", "dividend_yield",
    "net_mf_amount", "large_net_in", "margin_balance", "margin_delta_5d",
    "block_premium", "holder_change",
    "index_ret_1d", "index_ret_5d", "sector_ret_5d", "vix",
}

FUNCTION_ARITY: dict[str, int] = {
    "rank": 1, "zscore": 1, "demean": 1, "abs": 1, "sign": 1, "log": 1, "sqrt": 1,
    "ts_mean": 2, "ts_std": 2, "ts_max": 2, "ts_min": 2, "ts_sum": 2, "ts_rank": 2,
    "ts_argmax": 2, "ts_argmin": 2, "ts_count": 2, "ts_skew": 2, "ts_kurt": 2, "ts_product": 2,
    "delta": 2, "pct_change": 2, "shift": 2, "decay_linear": 2, "decay_exp": 2,
    "power": 2, "clamp": 3, "if_then": 3,
    "ts_corr": 3, "ts_cov": 3, "max": 2, "min": 2,
    # "quantile": 2, "neutralize": 2,  # not yet implemented
}


# ---------------------------------------------------------------------------
# AST nodes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ASTNode:
    """Base class for all AST nodes."""


@dataclass(frozen=True)
class Literal(ASTNode):
    value: float


@dataclass(frozen=True)
class Feature(ASTNode):
    name: str


@dataclass(frozen=True)
class FunctionCall(ASTNode):
    name: str
    args: list[ASTNode] = field(default_factory=list)


@dataclass(frozen=True)
class BinOp(ASTNode):
    op: str
    left: ASTNode
    right: ASTNode


@dataclass(frozen=True)
class UnaryOp(ASTNode):
    op: str
    operand: ASTNode


# ---------------------------------------------------------------------------
# Tokeniser
# ---------------------------------------------------------------------------

class TokenType(Enum):
    NUMBER = auto()
    IDENT = auto()
    PLUS = auto()
    MINUS = auto()
    STAR = auto()
    SLASH = auto()
    LPAREN = auto()
    RPAREN = auto()
    COMMA = auto()
    EOF = auto()


@dataclass
class Token:
    type: TokenType
    value: str
    pos: int


# Regex patterns ordered so that longer matches are tried first.
_TOKEN_SPEC: list[tuple[TokenType, str]] = [
    (TokenType.NUMBER, r"(?:\d+\.?\d*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?)"),
    (TokenType.IDENT,  r"[A-Za-z_][A-Za-z0-9_]*"),
    (TokenType.PLUS,   r"\+"),
    (TokenType.MINUS,  r"-"),
    (TokenType.STAR,   r"\*"),
    (TokenType.SLASH,  r"/"),
    (TokenType.LPAREN, r"\("),
    (TokenType.RPAREN, r"\)"),
    (TokenType.COMMA,  r","),
]
_MASTER_RE = re.compile(
    "|".join(f"(?P<T{tt.name}>{pat})" for tt, pat in _TOKEN_SPEC)
)


def _tokenize(src: str) -> list[Token]:
    tokens: list[Token] = []
    for m in _MASTER_RE.finditer(src):
        for tt, _ in _TOKEN_SPEC:
            val = m.group(f"T{tt.name}")
            if val is not None:
                tokens.append(Token(tt, val, m.start()))
                break
    tokens.append(Token(TokenType.EOF, "", len(src)))
    return tokens


# ---------------------------------------------------------------------------
# Pratt parser
# ---------------------------------------------------------------------------

class DSLParseError(Exception):
    pass


class _Parser:
    """Pratt (top-down operator precedence) parser."""

    # Binding powers for infix operators (left bp, right bp).
    _INFIX_BP: dict[TokenType, tuple[int, int]] = {
        TokenType.PLUS:  (1, 2),
        TokenType.MINUS: (1, 2),
        TokenType.STAR:  (3, 4),
        TokenType.SLASH: (3, 4),
    }

    _OP_MAP: dict[TokenType, str] = {
        TokenType.PLUS: "+",
        TokenType.MINUS: "-",
        TokenType.STAR: "*",
        TokenType.SLASH: "/",
    }

    def __init__(self, tokens: list[Token], source: str) -> None:
        self._tokens = tokens
        self._source = source
        self._pos = 0

    # -- helpers -------------------------------------------------------------

    def _peek(self) -> Token:
        return self._tokens[self._pos]

    def _advance(self) -> Token:
        tok = self._tokens[self._pos]
        self._pos += 1
        return tok

    def _expect(self, tt: TokenType) -> Token:
        tok = self._advance()
        if tok.type != tt:
            raise DSLParseError(
                f"Expected {tt.name} at position {tok.pos}, got {tok.type.name} ({tok.value!r})"
            )
        return tok

    # -- recursive descent with Pratt for infix ------------------------------

    def parse_expr(self, min_bp: int = 0) -> ASTNode:
        """Parse expression with minimum binding power *min_bp*."""
        lhs = self._parse_prefix()

        while True:
            tok = self._peek()
            if tok.type not in self._INFIX_BP:
                break
            l_bp, r_bp = self._INFIX_BP[tok.type]
            if l_bp < min_bp:
                break
            op_tok = self._advance()
            rhs = self.parse_expr(r_bp)
            lhs = BinOp(op=self._OP_MAP[op_tok.type], left=lhs, right=rhs)

        return lhs

    def _parse_prefix(self) -> ASTNode:
        tok = self._peek()

        # Unary minus
        if tok.type == TokenType.MINUS:
            self._advance()
            operand = self._parse_prefix()  # right-recursive for unary
            return UnaryOp(op="-", operand=operand)

        # Parenthesised sub-expression
        if tok.type == TokenType.LPAREN:
            self._advance()
            node = self.parse_expr(0)
            self._expect(TokenType.RPAREN)
            return node

        # Number
        if tok.type == TokenType.NUMBER:
            self._advance()
            return Literal(value=float(tok.value))

        # Identifier — could be a function call or a bare feature name
        if tok.type == TokenType.IDENT:
            self._advance()
            if self._peek().type == TokenType.LPAREN:
                # Function call
                self._advance()  # consume '('
                args: list[ASTNode] = []
                if self._peek().type != TokenType.RPAREN:
                    args.append(self.parse_expr(0))
                    while self._peek().type == TokenType.COMMA:
                        self._advance()  # consume ','
                        args.append(self.parse_expr(0))
                self._expect(TokenType.RPAREN)
                return FunctionCall(name=tok.value, args=args)
            else:
                return Feature(name=tok.value)

        raise DSLParseError(
            f"Unexpected token {tok.type.name} ({tok.value!r}) at position {tok.pos}"
        )


# ---------------------------------------------------------------------------
# Validation pass
# ---------------------------------------------------------------------------

def _ast_depth(node: ASTNode) -> int:
    """Return the *function-nesting* depth of the AST.

    Only FunctionCall nodes increase the depth counter.  BinOp and UnaryOp
    are transparent — they propagate the max depth of their children without
    adding a level.  This means ``rank(ret_5d) * rank(-pct_change(volume, 5))``
    has depth 2 (rank -> pct_change), not 5.
    """
    if isinstance(node, (Literal, Feature)):
        return 0
    if isinstance(node, UnaryOp):
        return _ast_depth(node.operand)
    if isinstance(node, BinOp):
        return max(_ast_depth(node.left), _ast_depth(node.right))
    if isinstance(node, FunctionCall):
        if not node.args:
            return 1
        return 1 + max(_ast_depth(a) for a in node.args)
    raise DSLParseError(f"Unknown node type: {type(node)}")


# Functions whose *last* argument is a lookback window (integer).
_WINDOW_FUNCTIONS: set[str] = {
    "ts_mean", "ts_std", "ts_max", "ts_min", "ts_sum", "ts_rank",
    "ts_argmax", "ts_argmin", "ts_corr", "ts_cov",
    "ts_count", "ts_skew", "ts_kurt", "ts_product",
    "delta", "pct_change", "shift", "decay_linear", "decay_exp",
}


def _validate(node: ASTNode) -> None:
    """Walk the AST and enforce safety constraints."""
    if isinstance(node, FunctionCall):
        if node.name not in ALLOWED_FUNCTIONS:
            raise DSLParseError(f"Unknown function: {node.name!r}")

        # Validate function arity
        if node.name in FUNCTION_ARITY:
            expected = FUNCTION_ARITY[node.name]
            if len(node.args) != expected:
                raise DSLParseError(
                    f"Function {node.name!r} expects {expected} argument(s), "
                    f"got {len(node.args)}"
                )

        # Validate lookback window for time-series functions
        if node.name in _WINDOW_FUNCTIONS:
            if len(node.args) < 2:
                raise DSLParseError(
                    f"Function {node.name!r} requires at least 2 arguments "
                    f"(got {len(node.args)})"
                )
            window_arg = node.args[-1]
            if not isinstance(window_arg, Literal):
                raise DSLParseError(
                    f"Window argument for {node.name!r} must be a numeric literal"
                )
            w = int(window_arg.value)
            if w != window_arg.value:
                raise DSLParseError(
                    f"Window argument for {node.name!r} must be an integer"
                )
            if w not in ALLOWED_LOOKBACK_WINDOWS:
                raise DSLParseError(
                    f"Window {w} not in allowed set: {sorted(ALLOWED_LOOKBACK_WINDOWS)}"
                )

        # Recurse into arguments
        for arg in node.args:
            _validate(arg)

    elif isinstance(node, BinOp):
        _validate(node.left)
        _validate(node.right)

    elif isinstance(node, UnaryOp):
        _validate(node.operand)

    elif isinstance(node, Feature):
        # Validate feature against whitelist
        if node.name not in FEATURE_WHITELIST:
            raise DSLParseError(
                f"Unknown feature: {node.name!r}. "
                f"Allowed features: {sorted(FEATURE_WHITELIST)}"
            )

    elif isinstance(node, Literal):
        pass  # numeric literals are always valid

    else:
        raise DSLParseError(f"Unknown node type: {type(node)}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse(expression: str) -> ASTNode:
    """
    Parse a DSL expression string into a validated AST.

    Raises DSLParseError on syntax errors or safety-constraint violations.

    Examples
    --------
    >>> parse("rank(delta(close, 5))")
    FunctionCall(name='rank', args=[FunctionCall(name='delta', args=[Feature(name='close'), Literal(value=5.0)])])

    >>> parse("rank(ret_5d) * rank(-volume)")
    BinOp(op='*', left=FunctionCall(...), right=FunctionCall(...))
    """
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise DSLParseError(
            f"Expression too long ({len(expression)} chars, max {MAX_EXPRESSION_LENGTH})"
        )

    tokens = _tokenize(expression)
    parser = _Parser(tokens, expression)
    ast = parser.parse_expr(0)

    # Make sure we consumed all tokens
    if parser._peek().type != TokenType.EOF:
        tok = parser._peek()
        raise DSLParseError(
            f"Unexpected trailing token {tok.type.name} ({tok.value!r}) at position {tok.pos}"
        )

    # Depth check
    depth = _ast_depth(ast)
    if depth > MAX_AST_DEPTH:
        raise DSLParseError(
            f"Expression depth {depth} exceeds maximum of {MAX_AST_DEPTH}"
        )

    # Safety validation (function whitelist, window whitelist, etc.)
    _validate(ast)

    return ast

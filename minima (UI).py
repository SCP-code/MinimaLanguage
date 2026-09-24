#!/usr/bin/env python3
# Minima++ (Windows-safe): full interpreter with lexer, Pratt parser, AST, scopes,
# control flow, arrays & dicts, functions/closures, stdlib, REPL & CLI.
# Handles UTF-8 BOM and CRLF line endings.
#
# Language overview:
# - Statements end with ';'    Blocks use { ... }
# - Variables:  let x = 10;
# - Assignments: x = 3; a[i] = 9; m["k"] = 2;
# - If/else: if cond { ... } else { ... }
# - While:   while cond { ... }
# - For:     for i in A..B { ... }   # end exclusive (like Python range(A, B))
# - Arrays:  [1,2,3]    Dicts: { name: "Bob", age: 30 }
# - Operators: + - * / %  == != < <= > >=  && || !  (unary - also)
# - Functions: func f(a,b){ return a+b; }   f(1,2);
# - Built-ins: print, len, input, int, float, str, type
# - Comments: // line, # line, /* block */
#
# Usage:
#   python minima.py file.minima
#   python minima.py   # REPL
#
# On Windows, you can package:  pyinstaller --onefile --name minima minima.py

import sys
import os
import re
import queue
import threading
from dataclasses import dataclass
from typing import Any, List, Optional, Dict

# ---------------- Tokenizer ----------------

WHITESPACE = ' \t'
DIGITS = '0123456789'
IDENT_START = '_abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
IDENT_BODY = IDENT_START + DIGITS
ESCAPES = {'n': '\n', 't': '\t', '"': '"', "'": "'", '\\': '\\'}

KEYWORDS = {
    'let': 'LET',
    'func': 'FUNC',
    'return': 'RETURN',
    'if': 'IF',
    'else': 'ELSE',
    'while': 'WHILE',
    'for': 'FOR',
    'in': 'IN'
}

@dataclass
class Token:
    kind: str
    text: str
    line: int
    col: int

class Lexer:
    def __init__(self, source: str):
        # Windows-safe normalization
        if source.startswith("\ufeff"):
            source = source[1:]               # strip UTF-8 BOM
        source = source.replace('\r\n', '\n')  # normalize CRLF to LF
        source = source.replace('\r', '\n')    # normalize any stray CR
        self.src = source
        self.i = 0
        self.line = 1
        self.col = 1

    def peek(self, n=0) -> str:
        j = self.i + n
        return self.src[j] if j < len(self.src) else '\0'

    def advance(self) -> str:
        ch = self.peek()
        self.i += 1
        if ch == '\n':
            self.line += 1
            self.col = 1
        else:
            self.col += 1
        return ch

    def token(self, kind, text, line=None, col=None) -> Token:
        return Token(kind, text, self.line if line is None else line, self.col if col is None else col)

    def skip_space_and_comments(self):
        while True:
            # whitespace
            moved = False
            while self.peek() in WHITESPACE or self.peek() == '\n':
                moved = True
                self.advance()
            if moved:
                continue
            # // comment
            if self.peek() == '/' and self.peek(1) == '/':
                while self.peek() not in ('\n', '\0'):
                    self.advance()
                continue
            # # comment
            if self.peek() == '#':
                while self.peek() not in ('\n', '\0'):
                    self.advance()
                continue
            # /* block comment */
            if self.peek() == '/' and self.peek(1) == '*':
                self.advance(); self.advance()
                while not (self.peek() == '*' and self.peek(1) == '/'):
                    if self.peek() == '\0':
                        raise SyntaxError("Unterminated block comment")
                    self.advance()
                self.advance(); self.advance()
                continue
            break

    def read_number(self) -> Token:
        start_line, start_col = self.line, self.col
        s = ''
        while self.peek() in DIGITS:
            s += self.advance()
        if self.peek() == '.' and self.peek(1) in DIGITS:
            s += self.advance()
            while self.peek() in DIGITS:
                s += self.advance()
        return Token('NUMBER', s, start_line, start_col)

    def read_ident(self) -> Token:
        start_line, start_col = self.line, self.col
        s = ''
        while self.peek() in IDENT_BODY:
            s += self.advance()
        if s in KEYWORDS:
            return Token(KEYWORDS[s], s, start_line, start_col)
        return Token('IDENT', s, start_line, start_col)

    def read_string(self) -> Token:
        start_line, start_col = self.line, self.col
        quote = self.advance()  # consume opening " or '
        out = ''
        while True:
            ch = self.advance()
            if ch == '\0':
                raise SyntaxError(f"Unterminated string at {start_line}:{start_col}")
            if ch == quote:
                break
            if ch == '\\':
                esc = self.advance()
                out += ESCAPES.get(esc, esc)
            else:
                out += ch
        return Token('STRING', out, start_line, start_col)

    def next(self) -> Token:
        self.skip_space_and_comments()
        ch = self.peek()
        if ch == '\0':
            return Token('EOF', '', self.line, self.col)
        if ch in DIGITS:
            return self.read_number()
        if ch in IDENT_START:
            return self.read_ident()
        if ch in ("'", '"'):
            return self.read_string()

        start_line, start_col = self.line, self.col
        two = ch + self.peek(1)
        if two in ('==','!=','<=','>=','&&','||'):
            self.advance(); self.advance()
            return Token(two, two, start_line, start_col)
        if two == '..':
            self.advance(); self.advance()
            return Token('RANGE', '..', start_line, start_col)

        self.advance()
        single_map = {
            '+': '+', '-': '-', '*': '*', '/': '/', '%': '%',
            '(': '(', ')': ')', '{': '{', '}': '}', '[': '[', ']': ']',
            ',': ',', ';': ';', ':': ':',
            '<': '<', '>': '>', '=': '=', '!': '!'
        }
        if ch in single_map:
            return Token(single_map[ch], ch, start_line, start_col)
        raise SyntaxError(f"Unexpected character '{ch}' at {start_line}:{start_col}")

# ---------------- Parser (Pratt) ----------------

PRECEDENCE = {
    '||': 1,
    '&&': 2,
    '==': 3, '!=': 3,
    '<': 4, '<=': 4, '>': 4, '>=': 4,
    '+': 5, '-': 5,
    '*': 6, '/': 6, '%': 6,
}

@dataclass
class AST: ...
@dataclass
class Number(AST): value: float
@dataclass
class String(AST): value: str
@dataclass
class Var(AST):    name: str
@dataclass
class Array(AST):  elements: List['AST']
@dataclass
class DictLit(AST): pairs: List[tuple]
@dataclass
class Unary(AST):  op: str; expr: 'AST'
@dataclass
class Binary(AST): left: 'AST'; op: str; right: 'AST'
@dataclass
class Call(AST):   callee: 'AST'; args: List['AST']
@dataclass
class Index(AST):  target: 'AST'; index: 'AST'
@dataclass
class Assign(AST): target: 'AST'; expr: 'AST'
@dataclass
class Let(AST):    name: str; expr: Optional['AST']
@dataclass
class Block(AST):  statements: List['AST']
@dataclass
class If(AST):     cond: 'AST'; then_b: Block; else_b: Optional[Block]
@dataclass
class While(AST):  cond: 'AST'; body: Block
@dataclass
class ForRange(AST): var: str; start: 'AST'; end: 'AST'; body: Block
@dataclass
class FuncDef(AST): name: str; params: List[str]; body: Block
@dataclass
class Return(AST): expr: Optional['AST']
@dataclass
class ExprStmt(AST): expr: 'AST'

class Parser:
    def __init__(self, tokens: List[Token]):
        self.toks = tokens
        self.pos = 0

    def peek(self) -> Token: return self.toks[self.pos]
    def advance(self) -> Token:
        t = self.toks[self.pos]; self.pos += 1; return t
    def match(self, *kinds) -> bool:
        if self.peek().kind in kinds:
            self.advance(); return True
        return False
    def expect(self, kind, msg):
        t = self.peek()
        if t.kind != kind:
            raise SyntaxError(f"{msg} at {t.line}:{t.col}, found {t.kind} '{t.text}'")
        return self.advance()

    def parse(self) -> Block:
        stmts = []
        while self.peek().kind != 'EOF':
            stmts.append(self.statement())
        return Block(stmts)

    def statement(self) -> AST:
        t = self.peek()
        if t.kind == 'LET':
            self.advance()
            name = self.expect('IDENT', "Expected variable name").text
            expr = None
            if self.match('='):
                expr = self.expression()
            self.expect(';', "Expected ';' after declaration")
            return Let(name, expr)
        if t.kind == 'FUNC':
            return self.func_def()
        if t.kind == 'IF':
            return self.if_stmt()
        if t.kind == 'WHILE':
            return self.while_stmt()
        if t.kind == 'FOR':
            return self.for_stmt()
        if t.kind == 'RETURN':
            self.advance()
            if self.match(';'):
                return Return(None)
            expr = self.expression()
            self.expect(';', "Expected ';' after return")
            return Return(expr)
        if t.kind == '{':
            return self.block()

        expr = self.expression()
        self.expect(';', "Expected ';' after statement")
        if isinstance(expr, Binary) and expr.op == '=' and isinstance(expr.left, (Var, Index)):
            return Assign(expr.left, expr.right)
        return ExprStmt(expr)

    def block(self) -> Block:
        self.expect('{', "Expected '{'")
        stmts = []
        while self.peek().kind != '}' and self.peek().kind != 'EOF':
            stmts.append(self.statement())
        self.expect('}', "Expected '}'")
        return Block(stmts)

    def func_def(self) -> FuncDef:
        self.expect('FUNC', "Expected 'func'")
        name = self.expect('IDENT', "Expected function name").text
        self.expect('(', "Expected '('")
        params = []
        if self.peek().kind != ')':
            while True:
                params.append(self.expect('IDENT', "Expected parameter").text)
                if not self.match(','): break
        self.expect(')', "Expected ')'")
        body = self.block()
        return FuncDef(name, params, body)

    def if_stmt(self) -> If:
        self.expect('IF', "Expected 'if'")
        if self.match('('):
            cond = self.expression(); self.expect(')', "Expected ')'")
        else:
            cond = self.expression()
        then_b = self.block()
        else_b = None
        if self.match('ELSE'):
            else_b = self.block()
        return If(cond, then_b, else_b)

    def while_stmt(self) -> While:
        self.expect('WHILE', "Expected 'while'")
        if self.match('('):
            cond = self.expression(); self.expect(')', "Expected ')'")
        else:
            cond = self.expression()
        body = self.block()
        return While(cond, body)

    def for_stmt(self) -> ForRange:
        self.expect('FOR', "Expected 'for'")
        var = self.expect('IDENT', "Expected loop variable").text
        self.expect('IN', "Expected 'in'")
        start = self.expression()
        self.expect('RANGE', "Expected '..'")
        end = self.expression()
        body = self.block()
        return ForRange(var, start, end, body)

    # -------- Expressions (Pratt) --------
    def expression(self, prec=0) -> AST:
        expr = self.prefix()
        while True:
            t = self.peek()
            if t.kind == '=':
                self.advance()
                right = self.expression(0)
                expr = Binary(expr, '=', right)
                continue
            if t.kind in PRECEDENCE and PRECEDENCE[t.kind] >= prec:
                op = t.kind; self.advance()
                right = self.expression(PRECEDENCE[op] + 1)
                expr = Binary(expr, op, right)
                continue
            if t.kind == '(':
                expr = self.finish_call(expr); continue
            if t.kind == '[':
                self.advance()
                idx = self.expression()
                self.expect(']', "Expected ']'")
                expr = Index(expr, idx); continue
            break
        return expr

    def prefix(self) -> AST:
        t = self.peek()
        if t.kind == 'NUMBER':
            self.advance()
            s = t.text
            if '.' in s: 
                return Number(float(s))
            return Number(int(s))
        if t.kind == 'STRING':
            self.advance(); return String(t.text)
        if t.kind == 'IDENT':
            self.advance(); return Var(t.text)
        if t.kind in ('-','!'):
            self.advance()
            return Unary(t.kind, self.expression(7))
        if t.kind == '(':
            self.advance()
            e = self.expression()
            self.expect(')', "Expected ')'")
            return e
        if t.kind == '[':
            self.advance()
            elems = []
            if self.peek().kind != ']':
                while True:
                    elems.append(self.expression())
                    if not self.match(','): break
            self.expect(']', "Expected ']'")
            return Array(elems)
        if t.kind == '{':
            self.advance()
            pairs = []
            if self.peek().kind != '}':
                while True:
                    # key: string or identifier
                    kt = self.peek()
                    if kt.kind == 'STRING':
                        key = String(self.advance().text)
                    elif kt.kind == 'IDENT':
                        key = String(self.advance().text)
                    else:
                        raise SyntaxError(f"Expected key at {kt.line}:{kt.col}")
                    self.expect(':', "Expected ':' after key")
                    val = self.expression()
                    pairs.append((key, val))
                    if not self.match(','): break
            self.expect('}', "Expected '}'")
            return DictLit(pairs)
        raise SyntaxError(f"Unexpected token {t.kind} '{t.text}' at {t.line}:{t.col}")

    def finish_call(self, callee: AST) -> AST:
        self.expect('(', "Expected '('")
        args = []
        if self.peek().kind != ')':
            while True:
                args.append(self.expression())
                if not self.match(','): break
        self.expect(')', "Expected ')'")
        return Call(callee, args)

# ---------------- Interpreter ----------------

class ReturnSignal(Exception):
    def __init__(self, value): self.value = value

class Env:
    def __init__(self, outer: Optional['Env']=None):
        self.vals: Dict[str, Any] = {}
        self.outer = outer
    def define(self, name, val): self.vals[name] = val
    def assign(self, name, val):
        if name in self.vals: self.vals[name] = val; return
        if self.outer: self.outer.assign(name, val); return
        raise RuntimeError(f"Undefined variable '{name}'")
    def get(self, name):
        if name in self.vals: return self.vals[name]
        if self.outer: return self.outer.get(name)
        raise RuntimeError(f"Undefined variable '{name}'")

class Function:
    def __init__(self, name: str, params: List[str], body: Block, closure: Env):
        self.name, self.params, self.body, self.closure = name, params, body, closure
    def __call__(self, interp, args: List[Any]):
        if len(args) != len(self.params):
            raise RuntimeError(f"Function {self.name} expects {len(self.params)} args, got {len(args)}")
        local = Env(self.closure)
        for p,a in zip(self.params, args): local.define(p, a)
        try:
            interp.exec_block(self.body, local)
        except ReturnSignal as rs:
            return rs.value
        return None

class ExecutionStopped(Exception):
    """Raised internally to unwind execution when the GUI's Stop button is used."""
    pass

class Interpreter:
    def __init__(self, output_fn=print, input_fn=input, stop_flag=None):
        self.output_fn = output_fn      # how print() writes output (GUI redirects this)
        self.input_fn = input_fn        # how input() gets a value (GUI redirects this)
        self.stop_flag = stop_flag      # optional threading.Event checked inside loops
        self.globals = Env()
        self.install_builtins(self.globals)

    def check_stop(self):
        if self.stop_flag is not None and self.stop_flag.is_set():
            raise ExecutionStopped()

    def install_builtins(self, env: Env):
        env.define('print', lambda *xs: self.output_fn(*xs))
        env.define('len', lambda x: len(x))
        env.define('input', lambda prompt="": self.input_fn(prompt))
        env.define('int', lambda x: int(x))
        env.define('float', lambda x: float(x))
        env.define('str', lambda x: str(x))
        env.define('type', lambda x: type(x).__name__)

    def truthy(self, v): return bool(v)

    def apply_bin(self, op, a, b):
        if op == '+': return a + b
        if op == '-': return a - b
        if op == '*': return a * b
        if op == '/': return a / b
        if op == '%': return a % b
        if op == '==': return a == b
        if op == '!=': return a != b
        if op == '<': return a < b
        if op == '<=': return a <= b
        if op == '>': return a > b
        if op == '>=': return a >= b
        raise RuntimeError(f"Unknown operator {op}")

    def eval(self, node: AST, env: Env):
        if isinstance(node, Number): return node.value
        if isinstance(node, String): return node.value
        if isinstance(node, Var):    return env.get(node.name)
        if isinstance(node, Array):  return [self.eval(e, env) for e in node.elements]
        if isinstance(node, DictLit): return { self.eval(k, env): self.eval(v, env) for (k,v) in node.pairs }
        if isinstance(node, Unary):
            v = self.eval(node.expr, env)
            if node.op == '-': return -v
            if node.op == '!': return not self.truthy(v)
            raise RuntimeError(f"Unknown unary {node.op}")
        if isinstance(node, Binary):
            if node.op == '=':
                if isinstance(node.left, Var):
                    val = self.eval(node.right, env)
                    env.assign(node.left.name, val); return val
                if isinstance(node.left, Index):
                    tgt = self.eval(node.left.target, env)
                    idx = self.eval(node.left.index, env)
                    val = self.eval(node.right, env)
                    tgt[idx] = val; return val
                raise RuntimeError("Invalid assignment target")
            left = self.eval(node.left, env)
            if node.op == '||':
                return left or self.eval(node.right, env)
            if node.op == '&&':
                return left and self.eval(node.right, env)
            right = self.eval(node.right, env)
            return self.apply_bin(node.op, left, right)
        if isinstance(node, Call):
            cal = self.eval(node.callee, env)
            args = [self.eval(a, env) for a in node.args]
            if isinstance(cal, Function):
                return cal(self, args)
            if callable(cal):
                return cal(*args)
            raise RuntimeError("Attempted to call a non-function")
        if isinstance(node, Index):
            tgt = self.eval(node.target, env)
            idx = self.eval(node.index, env)
            return tgt[idx]
        if isinstance(node, Assign):
            # handled as Binary '='
            val = self.eval(node.expr, env)
            if isinstance(node.target, Var):
                env.assign(node.target.name, val)
            elif isinstance(node.target, Index):
                container = self.eval(node.target.target, env)
                idx = self.eval(node.target.index, env)
                container[idx] = val
            return val
        if isinstance(node, ExprStmt):
            return self.eval(node.expr, env)
        if isinstance(node, Let):
            val = self.eval(node.expr, env) if node.expr is not None else None
            env.define(node.name, val); return None
        if isinstance(node, Block):
            self.exec_block(node, Env(env)); return None
        if isinstance(node, If):
            if self.truthy(self.eval(node.cond, env)):
                self.exec_block(node.then_b, Env(env))
            elif node.else_b:
                self.exec_block(node.else_b, Env(env))
            return None
        if isinstance(node, While):
            while self.truthy(self.eval(node.cond, env)):
                self.check_stop()
                self.exec_block(node.body, Env(env))
            return None
        if isinstance(node, ForRange):
            a = self.eval(node.start, env)
            b = self.eval(node.end, env)
            step = 1 if b >= a else -1
            for v in range(a, b, step):
                self.check_stop()
                loop = Env(env); loop.define(node.var, v)
                self.exec_block(node.body, loop)
            return None
        if isinstance(node, FuncDef):
            fn = Function(node.name, node.params, node.body, env)
            env.define(node.name, fn); return None
        if isinstance(node, Return):
            val = self.eval(node.expr, env) if node.expr is not None else None
            raise ReturnSignal(val)
        raise RuntimeError(f"Unknown AST node {node}")

    def exec_block(self, block: Block, env: Env):
        for st in block.statements:
            self.check_stop()
            self.eval(st, env)

# --------------- Frontend helpers ---------------

def tokenize(source: str) -> List[Token]:
    lex = Lexer(source)
    out: List[Token] = []
    while True:
        t = lex.next(); out.append(t)
        if t.kind == 'EOF': break
    return out

def parse(source: str) -> Block:
    tokens = tokenize(source)
    parser = Parser(tokens)
    return parser.parse()

def run_source(source: str, interp: Optional[Interpreter]=None):
    if interp is None: interp = Interpreter()
    ast = parse(source)
    return interp.eval(ast, interp.globals)

def run_file(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        src = f.read()
    run_source(src)

def repl():
    print("Minima++ REPL — Windows-safe. End statements with ';'. Type 'exit' on empty prompt to quit.")
    interp = Interpreter()
    buf = ""
    while True:
        try:
            line = input(">>> " if not buf else "... ")
        except EOFError:
            print(); break
        if not buf and line.strip().lower() == 'exit':
            break
        buf += line + "\n"
        # Heuristic: try to run when a statement likely ends
        if line.strip().endswith(';') or line.strip() == '}' or line.strip() == '':
            try:
                run_source(buf, interp)
            except Exception as e:
                print("Error:", e)
            buf = ""

# --------------- GUI (Tkinter IDE) ---------------

APP_TITLE = "Minima++ IDE"

def launch_gui(initial_path: Optional[str] = None):
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, simpledialog, font as tkfont
    except ImportError:
        print("Tkinter isn't available in this Python install.")
        print("On Debian/Ubuntu, try: sudo apt-get install python3-tk")
        sys.exit(1)

    # Highlighting patterns, built from the language's own keyword table.
    KEYWORD_RE = re.compile(r'\b(?:' + '|'.join(re.escape(k) for k in KEYWORDS) + r')\b')
    NUMBER_RE = re.compile(r'\b\d+(?:\.\d+)?\b')
    STRING_RE = re.compile(r'"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'')
    LINE_COMMENT_RE = re.compile(r'//[^\n]*|#[^\n]*')
    BLOCK_COMMENT_RE = re.compile(r'/\*.*?\*/', re.DOTALL)

    class MinimaGUI:
        def __init__(self, root):
            self.root = root
            self.filepath = None
            self.msg_queue = queue.Queue()
            self.stop_flag = threading.Event()
            self.worker = None
            self.mono = tkfont.Font(family="Consolas", size=11)
            if self.mono.actual('family').lower() != 'consolas':
                self.mono = tkfont.Font(family="Courier New", size=11)

            root.title(APP_TITLE)
            root.geometry("1050x720")
            root.protocol("WM_DELETE_WINDOW", self.on_close)

            self._build_menu()
            self._build_layout()
            self._bind_shortcuts()
            self._update_title()
            self._poll_queue()

            if initial_path:
                self._load_path(initial_path)

        # ---- menu / shortcuts ----
        def _build_menu(self):
            mb = tk.Menu(self.root)
            filem = tk.Menu(mb, tearoff=0)
            filem.add_command(label="New", accelerator="Ctrl+N", command=self.new_file)
            filem.add_command(label="Open...", accelerator="Ctrl+O", command=self.open_file)
            filem.add_command(label="Save", accelerator="Ctrl+S", command=self.save_file)
            filem.add_command(label="Save As...", accelerator="Ctrl+Shift+S", command=self.save_file_as)
            filem.add_separator()
            filem.add_command(label="Exit", accelerator="Ctrl+Q", command=self.on_close)
            mb.add_cascade(label="File", menu=filem)

            runm = tk.Menu(mb, tearoff=0)
            runm.add_command(label="Run", accelerator="F5", command=self.run_code)
            runm.add_command(label="Stop", accelerator="Ctrl+.", command=self.stop_code)
            runm.add_separator()
            runm.add_command(label="Clear Console", command=self.clear_output)
            mb.add_cascade(label="Run", menu=runm)

            helpm = tk.Menu(mb, tearoff=0)
            helpm.add_command(label="Language Quick Reference", command=self.show_help)
            helpm.add_command(label="About", command=self.show_about)
            mb.add_cascade(label="Help", menu=helpm)

            self.root.config(menu=mb)

        def _bind_shortcuts(self):
            self.root.bind('<Control-n>', lambda e: self.new_file())
            self.root.bind('<Control-o>', lambda e: self.open_file())
            self.root.bind('<Control-s>', lambda e: self.save_file())
            self.root.bind('<Control-Shift-S>', lambda e: self.save_file_as())
            self.root.bind('<F5>', lambda e: self.run_code())
            self.root.bind('<Control-period>', lambda e: self.stop_code())
            self.root.bind('<Control-q>', lambda e: self.on_close())

        # ---- layout ----
        def _build_layout(self):
            toolbar = tk.Frame(self.root)
            toolbar.pack(fill=tk.X, side=tk.TOP)
            tk.Button(toolbar, text="\u25b6 Run  (F5)", command=self.run_code).pack(side=tk.LEFT, padx=4, pady=3)
            tk.Button(toolbar, text="\u25a0 Stop", command=self.stop_code).pack(side=tk.LEFT, padx=4, pady=3)
            tk.Button(toolbar, text="Clear Console", command=self.clear_output).pack(side=tk.LEFT, padx=4, pady=3)

            status = tk.Frame(self.root, relief=tk.SUNKEN, bd=1)
            status.pack(fill=tk.X, side=tk.BOTTOM)
            self.status_left = tk.Label(status, text="Ready", anchor='w')
            self.status_left.pack(side=tk.LEFT, padx=6)
            self.status_right = tk.Label(status, text="Ln 1, Col 1", anchor='e')
            self.status_right.pack(side=tk.RIGHT, padx=6)

            paned = tk.PanedWindow(self.root, orient=tk.VERTICAL, sashwidth=6, sashrelief=tk.RAISED)
            paned.pack(fill=tk.BOTH, expand=True)

            editor_frame = tk.Frame(paned)
            self.linenumbers = tk.Text(editor_frame, width=4, padx=6, border=0,
                                        state='disabled', takefocus=0, font=self.mono,
                                        background="#eef0f2", foreground="#888888")
            self.linenumbers.pack(side=tk.LEFT, fill=tk.Y)

            self.editor = tk.Text(editor_frame, wrap='none', undo=True, font=self.mono,
                                   background="#ffffff", insertbackground="#000000")
            self.editor.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

            vsb = tk.Scrollbar(editor_frame, orient=tk.VERTICAL, command=self._on_scroll)
            vsb.pack(side=tk.RIGHT, fill=tk.Y)
            self.editor.config(yscrollcommand=vsb.set)

            self.editor.tag_config('keyword', foreground='#0000cc')
            self.editor.tag_config('string', foreground='#a31515')
            self.editor.tag_config('comment', foreground='#008000')
            self.editor.tag_config('number', foreground='#098658')

            self.editor.bind('<KeyRelease>', self._on_edit)
            self.editor.bind('<ButtonRelease-1>', self._update_cursor_status)
            self.editor.bind('<MouseWheel>', self._on_mousewheel)
            self.editor.bind('<Button-4>', self._on_mousewheel)
            self.editor.bind('<Button-5>', self._on_mousewheel)
            self.editor.edit_modified(False)
            self.editor.bind('<<Modified>>', self._on_modified)

            paned.add(editor_frame, stretch="always", minsize=150)

            console_frame = tk.Frame(paned)
            tk.Label(console_frame, text="Console", anchor='w', bg="#2d2d2d", fg="white",
                     font=(self.mono.actual('family'), 9, 'bold'), padx=6).pack(fill=tk.X)
            self.console = tk.Text(console_frame, height=10, state='disabled', font=self.mono,
                                    background="#1e1e1e", foreground="#d4d4d4", wrap='word')
            self.console.pack(fill=tk.BOTH, expand=True)
            self.console.tag_config('error', foreground='#f14c4c')

            paned.add(console_frame, minsize=100)

            self._update_linenumbers()

        def _on_scroll(self, *args):
            self.editor.yview(*args)
            self.linenumbers.yview(*args)

        def _on_mousewheel(self, event):
            self.root.after_idle(lambda: self.linenumbers.yview_moveto(self.editor.yview()[0]))

        def _on_modified(self, event=None):
            self.editor.edit_modified(False)
            self._update_title()

        def _on_edit(self, event=None):
            self._update_linenumbers()
            self._highlight()
            self._update_cursor_status()

        def _update_cursor_status(self, event=None):
            row, col = self.editor.index(tk.INSERT).split('.')
            self.status_right.config(text=f"Ln {row}, Col {int(col) + 1}")

        def _update_linenumbers(self):
            n = int(self.editor.index('end-1c').split('.')[0])
            text = '\n'.join(str(i) for i in range(1, n + 1))
            self.linenumbers.config(state='normal')
            self.linenumbers.delete('1.0', 'end')
            self.linenumbers.insert('1.0', text)
            self.linenumbers.config(state='disabled')
            self.linenumbers.yview_moveto(self.editor.yview()[0])

        def _highlight(self):
            content = self.editor.get('1.0', 'end-1c')
            for tag in ('keyword', 'string', 'comment', 'number'):
                self.editor.tag_remove(tag, '1.0', 'end')

            def at(offset):
                return f"1.0+{offset}c"

            spans = []
            for m in STRING_RE.finditer(content):
                self.editor.tag_add('string', at(m.start()), at(m.end()))
                spans.append((m.start(), m.end()))

            def in_string(pos):
                return any(s <= pos < e for s, e in spans)

            for m in KEYWORD_RE.finditer(content):
                if not in_string(m.start()):
                    self.editor.tag_add('keyword', at(m.start()), at(m.end()))
            for m in NUMBER_RE.finditer(content):
                if not in_string(m.start()):
                    self.editor.tag_add('number', at(m.start()), at(m.end()))
            for m in BLOCK_COMMENT_RE.finditer(content):
                self.editor.tag_add('comment', at(m.start()), at(m.end()))
            for m in LINE_COMMENT_RE.finditer(content):
                if not in_string(m.start()):
                    self.editor.tag_add('comment', at(m.start()), at(m.end()))

        # ---- file operations ----
        def _title_name(self):
            return os.path.basename(self.filepath) if self.filepath else "Untitled"

        def _update_title(self):
            mark = "*" if self.editor.edit_modified() else ""
            self.root.title(f"{mark}{self._title_name()} \u2014 {APP_TITLE}")

        def _confirm_discard_changes(self):
            if not self.editor.edit_modified():
                return True
            resp = messagebox.askyesnocancel("Unsaved changes", "Save changes before continuing?")
            if resp is None:
                return False
            if resp:
                return self.save_file()
            return True

        def new_file(self):
            if not self._confirm_discard_changes():
                return
            self.editor.delete('1.0', 'end')
            self.filepath = None
            self.editor.edit_modified(False)
            self._on_edit()
            self._update_title()

        def open_file(self):
            if not self._confirm_discard_changes():
                return
            path = filedialog.askopenfilename(filetypes=[("Minima files", "*.minima"), ("All files", "*.*")])
            if path:
                self._load_path(path)

        def _load_path(self, path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    src = f.read()
            except OSError as e:
                messagebox.showerror("Open failed", str(e))
                return
            self.editor.delete('1.0', 'end')
            self.editor.insert('1.0', src)
            self.filepath = path
            self.editor.edit_modified(False)
            self._on_edit()
            self._update_title()

        def save_file(self):
            if self.filepath is None:
                return self.save_file_as()
            try:
                with open(self.filepath, 'w', encoding='utf-8') as f:
                    f.write(self.editor.get('1.0', 'end-1c'))
            except OSError as e:
                messagebox.showerror("Save failed", str(e))
                return False
            self.editor.edit_modified(False)
            self._update_title()
            return True

        def save_file_as(self):
            path = filedialog.asksaveasfilename(defaultextension=".minima",
                                                 filetypes=[("Minima files", "*.minima"), ("All files", "*.*")])
            if not path:
                return False
            self.filepath = path
            return self.save_file()

        def on_close(self):
            if self._confirm_discard_changes():
                self.root.destroy()

        # ---- run / stop ----
        def clear_output(self):
            self.console.config(state='normal')
            self.console.delete('1.0', 'end')
            self.console.config(state='disabled')

        def _append_console(self, text, tag=None):
            self.console.config(state='normal')
            self.console.insert('end', text, tag or ())
            self.console.see('end')
            self.console.config(state='disabled')

        def run_code(self):
            if self.worker and self.worker.is_alive():
                messagebox.showinfo("Already running", "Your program is already running. Use Stop first.")
                return
            source = self.editor.get('1.0', 'end-1c')
            self.clear_output()
            self.stop_flag.clear()
            self.status_left.config(text="Running\u2026")
            interp = Interpreter(output_fn=self._enqueue_output,
                                  input_fn=self._request_input,
                                  stop_flag=self.stop_flag)
            self.worker = threading.Thread(target=self._run_worker, args=(source, interp), daemon=True)
            self.worker.start()

        def stop_code(self):
            if self.worker and self.worker.is_alive():
                self.stop_flag.set()
                self.status_left.config(text="Stopping\u2026")
            else:
                self.status_left.config(text="Ready")

        def _run_worker(self, source, interp):
            try:
                ast_root = parse(source)
                interp.eval(ast_root, interp.globals)
                self.msg_queue.put(('status', 'Finished.'))
            except ExecutionStopped:
                self.msg_queue.put(('status', 'Stopped.'))
            except (SyntaxError, RuntimeError) as e:
                self.msg_queue.put(('error', f"{type(e).__name__}: {e}\n"))
                self.msg_queue.put(('status', 'Finished with errors.'))
            except Exception as e:
                self.msg_queue.put(('error', f"Internal error: {e}\n"))
                self.msg_queue.put(('status', 'Finished with errors.'))

        def _enqueue_output(self, *args):
            self.msg_queue.put(('output', ' '.join(str(a) for a in args) + '\n'))

        def _request_input(self, prompt=""):
            # Called from the worker thread; blocks until the main thread supplies a value
            # via a modal dialog, so it's safe to call Minima's input() from a running script.
            holder = {}
            event = threading.Event()
            self.msg_queue.put(('input', (prompt, holder, event)))
            event.wait()
            return holder.get('value', '')

        def _poll_queue(self):
            try:
                while True:
                    kind, payload = self.msg_queue.get_nowait()
                    if kind == 'output':
                        self._append_console(payload)
                    elif kind == 'error':
                        self._append_console(payload, 'error')
                    elif kind == 'status':
                        self.status_left.config(text=payload)
                    elif kind == 'input':
                        prompt, holder, event = payload
                        val = simpledialog.askstring("Input", prompt or "Input:", parent=self.root)
                        holder['value'] = val if val is not None else ''
                        event.set()
            except queue.Empty:
                pass
            self.root.after(40, self._poll_queue)

        def show_about(self):
            messagebox.showinfo("About", f"{APP_TITLE}\nA simple editor and runner for Minima++ scripts.")

        def show_help(self):
            messagebox.showinfo(
                "Quick Reference",
                "let x = 10;\n"
                "if cond { ... } else { ... }\n"
                "while cond { ... }\n"
                "for i in A..B { ... }\n"
                "func f(a,b){ return a+b; }\n"
                "Arrays: [1,2,3]   Dicts: { name: \"Bob\" }\n"
                "Built-ins: print, len, input, int, float, str, type"
            )

    root = tk.Tk()
    MinimaGUI(root)
    root.mainloop()

def main():
    args = sys.argv[1:]
    if '--repl' in args:
        repl()
    elif '--gui' in args:
        args.remove('--gui')
        launch_gui(args[0] if args else None)
    elif args:
        run_file(args[0])
    else:
        launch_gui()

if __name__ == '__main__':
    main()

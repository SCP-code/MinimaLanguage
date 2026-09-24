#!/usr/bin/env python3
# Minima Language + PyQt5 IDE (no external deps)
# - Interpreter: lexer, Pratt parser, AST, env/closures
# - Arrays (nested, indexing, slicing a[b:c])
# - Dicts, Strings, Numbers, Booleans, Null
# - Classes (class, new, init, fields, methods, this)
# - Control flow: if/elif/else, while, for-in, range A..B, return
# - Operators: + - * / % **, == != < <= > >=, && || !, unary -
# - Builtins: print, input, len, int, float, str, type, os
# - Comments: //, #, /* ... */
# - GUI: dark theme, editor, console, buttons

import sys, os as _pyos, subprocess, math, re
from dataclasses import dataclass
from typing import Any, List, Optional, Dict, Tuple

# ------------------------- Lexer -------------------------

WHITESPACE = ' \t'
DIGITS = '0123456789'
IDENT_START = '_abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
IDENT_BODY = IDENT_START + DIGITS
ESCAPES = {'n':'\n','t':'\t','r':'\r','"':'"',"\'":"\'",'\\':'\\'}

KEYWORDS = {
    'let':'LET','var':'LET',
    'func':'FUNC','return':'RETURN',
    'if':'IF','else':'ELSE','elif':'ELIF',
    'while':'WHILE','for':'FOR','in':'IN',
    'class':'CLASS','new':'NEW',
    'true':'TRUE','false':'FALSE','null':'NULL',
}

@dataclass
class Token:
    kind: str
    text: str
    line: int
    col: int

# Replace your current Lexer class with this one

class Lexer:
    def __init__(self, source: str):
        # normalize BOM and CRLF/CR to LF
        if source.startswith("\ufeff"):
            source = source[1:]
        source = source.replace('\r\n', '\n').replace('\r', '\n')
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

    def skip_space_comments(self):
        while True:
            moved = False
            # whitespace and newlines
            while self.peek() in WHITESPACE or self.peek() == '\n':
                self.advance(); moved = True
            if moved:
                continue
            # line comment // or #
            if (self.peek() == '/' and self.peek(1) == '/') or self.peek() == '#':
                while self.peek() not in ('\n', '\0'):
                    self.advance()
                continue
            # block comment /* ... */
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
        quote = self.advance()  # consume opening quote
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

    # Public method used by older code: lex.next()
    def next(self) -> Token:
        return self.next_token()

    # Python iterator protocol support: next(lex) -> calls __next__
    def __iter__(self):
        return self

    def __next__(self):
        tok = self.next_token()
        if tok.kind == 'EOF':
            raise StopIteration
        return tok

    # The actual token-producing implementation
    def next_token(self) -> Token:
        self.skip_space_comments()
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
        if two in ('==', '!=', '<=', '>=', '&&', '||', '**'):
            self.advance(); self.advance()
            return Token(two, two, start_line, start_col)

        if two == '..':
            self.advance(); self.advance()
            return Token('RANGE', '..', start_line, start_col)

        # single-character tokens: use the actual character as the kind
        self.advance()
        single_map = {
            '+': '+', '-': '-', '*': '*', '/': '/', '%': '%',
            '(': '(', ')': ')', '{': '{', '}': '}', '[': '[', ']': ']',
            ',': ',', ';': ';', ':': ':', '.': '.',
            '<': '<', '>': '>', '=': '=', '!': '!'
        }
        if ch in single_map:
            return Token(single_map[ch], ch, start_line, start_col)

        raise SyntaxError(f"Unexpected character '{ch}' at {start_line}:{start_col}")



# ------------------------- AST -------------------------

@dataclass
class AST: ...
@dataclass
class Number(AST): value: float
@dataclass
class String(AST): value: str
@dataclass
class Bool(AST): value: bool
@dataclass
class Null(AST): ...
@dataclass
class Var(AST): name: str
@dataclass
class Array(AST): elements: List['AST']
@dataclass
class DictLit(AST): pairs: List[Tuple['AST','AST']]
@dataclass
class Unary(AST): op: str; expr: 'AST'
@dataclass
class Binary(AST): left: 'AST'; op: str; right: 'AST'
@dataclass
class Call(AST): callee: 'AST'; args: List['AST']
@dataclass
class Index(AST): target: 'AST'; index: Optional['AST']; slice_end: Optional['AST']
@dataclass
class Member(AST): target: 'AST'; name: str
@dataclass
class Assign(AST): target: 'AST'; expr: 'AST'
@dataclass
class Let(AST): name: str; expr: Optional['AST']
@dataclass
class Block(AST): statements: List['AST']
@dataclass
class If(AST): cond: 'AST'; then_b: Block; elifs: List[Tuple['AST',Block]]; else_b: Optional[Block]
@dataclass
class While(AST): cond: 'AST'; body: Block
@dataclass
class ForIn(AST): var: str; iterable: 'AST'; body: Block
@dataclass
class ForRange(AST): var: str; start: 'AST'; end: 'AST'; body: Block
@dataclass
class FuncDef(AST): name: str; params: List[str]; body: Block
@dataclass
class Return(AST): expr: Optional['AST']
@dataclass
class ExprStmt(AST): expr: 'AST'
@dataclass
class ClassDef(AST): name: str; methods: Dict[str, 'FuncDef']

# ------------------------- Parser (Pratt) -------------------------

PRECEDENCE = {
    '||':1,
    '&&':2,
    '==':3,'!=':3,
    '<':4,'<=':4,'>':4,'>=':4,
    '+':5,'-':5,
    '*':6,'/':6,'%':6,
    '**':7,
}

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
        stmts=[]
        while self.peek().kind != 'EOF':
            stmts.append(self.statement())
        return Block(stmts)

    def statement(self) -> AST:
        t = self.peek()
        if t.kind == 'LET':
            self.advance()
            name = self.expect('IDENT',"Expected variable name").text
            expr=None
            if self.match('='): expr = self.expression()
            self.expect(';',"Expected ';' after declaration")
            return Let(name,expr)
        if t.kind == 'FUNC':
            return self.func_def()
        if t.kind == 'CLASS':
            return self.class_def()
        if t.kind == 'IF':
            return self.if_stmt()
        if t.kind == 'WHILE':
            return self.while_stmt()
        if t.kind == 'FOR':
            return self.for_stmt()
        if t.kind == 'RETURN':
            self.advance()
            if self.match(';'): return Return(None)
            expr = self.expression()
            self.expect(';',"Expected ';' after return")
            return Return(expr)
        if t.kind == '{':
            return self.block()
        # Expr statement
        expr = self.expression()
        self.expect(';',"Expected ';' after statement")
        if isinstance(expr, Binary) and expr.op == '=' and isinstance(expr.left, (Var, Index, Member)):
            return Assign(expr.left, expr.right)
        return ExprStmt(expr)

    def block(self) -> Block:
        self.expect('{',"Expected '{'")
        stmts=[]
        while self.peek().kind != '}' and self.peek().kind != 'EOF':
            stmts.append(self.statement())
        self.expect('}',"Expected '}'")
        return Block(stmts)

    def func_def(self) -> FuncDef:
        self.expect('FUNC',"Expected 'func'")
        name = self.expect('IDENT',"Expected function name").text
        self.expect('(',"Expected '('")
        params=[]
        if self.peek().kind != ')':
            while True:
                params.append(self.expect('IDENT',"Expected parameter").text)
                if not self.match(','): break
        self.expect(')',"Expected ')'")
        body = self.block()
        return FuncDef(name,params,body)

    def class_def(self) -> ClassDef:
        self.expect('CLASS',"Expected 'class'")
        name = self.expect('IDENT',"Expected class name").text
        self.expect('{',"Expected '{' after class name")
        methods: Dict[str,FuncDef] = {}
        while self.peek().kind != '}' and self.peek().kind != 'EOF':
            f = self.func_def()
            methods[f.name]=f
        self.expect('}',"Expected '}' after class body")
        return ClassDef(name, methods)

    def if_stmt(self) -> If:
        self.expect('IF',"Expected 'if'")
        if self.match('('):
            cond = self.expression(); self.expect(')',"Expected ')'")
        else:
            cond = self.expression()
        then_b = self.block()
        elifs=[]
        while self.match('ELIF'):
            if self.match('('):
                c = self.expression(); self.expect(')',"Expected ')'")
            else:
                c = self.expression()
            b = self.block()
            elifs.append((c,b))
        else_b=None
        if self.match('ELSE'):
            else_b = self.block()
        return If(cond,then_b,elifs,else_b)

    def while_stmt(self) -> While:
        self.expect('WHILE',"Expected 'while'")
        if self.match('('):
            cond = self.expression(); self.expect(')',"Expected ')'")
        else:
            cond = self.expression()
        body = self.block()
        return While(cond, body)

    def for_stmt(self) -> AST:
        self.expect('FOR',"Expected 'for'")
        var = self.expect('IDENT',"Expected loop variable").text
        self.expect('IN',"Expected 'in'")
        start_expr = self.expression()
        if self.match('RANGE'):
            end_expr = self.expression()
            body = self.block()
            return ForRange(var, start_expr, end_expr, body)
        else:
            iterable = start_expr
            body = self.block()
            return ForIn(var, iterable, body)

    # -------- Expressions --------
    def expression(self, prec=0) -> AST:
        expr = self.prefix()
        while True:
            t = self.peek()
            if t.kind == '=':
                self.advance()
                right = self.expression(0)
                expr = Binary(expr,'=',right); continue
            if t.kind in PRECEDENCE and PRECEDENCE[t.kind] >= prec:
                op = t.kind; self.advance()
                next_prec = PRECEDENCE[op] + (0 if op=='**' else 1)
                right = self.expression(next_prec)
                expr = Binary(expr,op,right); continue
            if t.kind == '(':
                expr = self.finish_call(expr); continue
            if t.kind == '[':
                self.advance()
                idx=None; end=None
                if self.peek().kind != ']':
                    if self.peek().kind != ':':
                        idx = self.expression()
                    if self.match(':'):
                        if self.peek().kind != ']':
                            end = self.expression()
                self.expect(']',"Expected ']'")
                expr = Index(expr, idx, end); continue
            if t.kind == '.':
                self.advance()
                name = self.expect('IDENT',"Expected member name after '.'").text
                expr = Member(expr, name); continue
            break
        return expr

    def prefix(self) -> AST:
        t = self.peek()
        if t.kind == 'NUMBER':
            self.advance()
            s=t.text
            if '.' in s: return Number(float(s))
            return Number(int(s))
        if t.kind == 'STRING':
            self.advance(); return String(t.text)
        if t.kind == 'TRUE': self.advance(); return Bool(True)
        if t.kind == 'FALSE': self.advance(); return Bool(False)
        if t.kind == 'NULL': self.advance(); return Null()
        if t.kind == 'IDENT':
            self.advance(); return Var(t.text)
        if t.kind == 'NEW':
            self.advance()
            cname = self.expect('IDENT',"Expected class name after 'new'").text
            self.expect('(',"Expected '('")
            args=[]
            if self.peek().kind != ')':
                while True:
                    args.append(self.expression())
                    if not self.match(','): break
            self.expect(')',"Expected ')'")
            # Represent as Call on special Var name; resolved in interpreter
            return Call(Var(f'__new__{cname}'), args)
        if t.kind in ('-','!'):
            self.advance(); return Unary(t.kind, self.expression(8))
        if t.kind == '(':
            self.advance()
            e = self.expression()
            self.expect(')',"Expected ')'")
            return e
        if t.kind == '[':
            self.advance()
            elems=[]
            if self.peek().kind != ']':
                while True:
                    elems.append(self.expression())
                    if not self.match(','): break
            self.expect(']',"Expected ']'")
            return Array(elems)
        if t.kind == '{':
            self.advance()
            pairs=[]
            if self.peek().kind != '}':
                while True:
                    kt = self.peek()
                    if kt.kind == 'STRING':
                        key = String(self.advance().text)
                    elif kt.kind == 'IDENT':
                        key = String(self.advance().text)
                    else:
                        raise SyntaxError(f"Expected key at {kt.line}:{kt.col}")
                    self.expect(':',"Expected ':' after key")
                    val = self.expression()
                    pairs.append((key,val))
                    if not self.match(','): break
            self.expect('}',"Expected '}'")
            return DictLit(pairs)
        raise SyntaxError(f"Unexpected token {t.kind} '{t.text}' at {t.line}:{t.col}")

    def finish_call(self, callee: AST) -> AST:
        self.expect('(',"Expected '('")
        args=[]
        if self.peek().kind != ')':
            while True:
                args.append(self.expression())
                if not self.match(','): break
        self.expect(')',"Expected ')'")
        return Call(callee,args)

# ------------------------- Interpreter -------------------------

class ReturnSignal(Exception):
    def __init__(self, value): self.value = value

class Env:
    def __init__(self, outer: Optional['Env']=None):
        self.vals: Dict[str, Any] = {}
        self.outer = outer
    def define(self, name, val): self.vals[name] = val
    def assign(self, name, val):
        if name in self.vals: self.vals[name]=val; return
        if self.outer: self.outer.assign(name,val); return
        raise RuntimeError(f"Undefined variable '{name}'")
    def get(self, name):
        if name in self.vals: return self.vals[name]
        if self.outer: return self.outer.get(name)
        raise RuntimeError(f"Undefined variable '{name}'")

class Function:
    def __init__(self, name: str, params: List[str], body: Block, closure: Env):
        self.name,self.params,self.body,self.closure = name,params,body,closure
    def __call__(self, interp, args: List[Any]):
        if len(args) != len(self.params):
            raise RuntimeError(f"Function {self.name} expects {len(self.params)} args, got {len(args)}")
        local = Env(self.closure)
        for p,a in zip(self.params,args): local.define(p,a)
        try:
            interp.exec_block(self.body, local)
        except ReturnSignal as rs:
            return rs.value
        return None

class Klass:
    def __init__(self, name: str, methods: Dict[str, Function]):
        self.name=name; self.methods=methods
    def instantiate(self, interp, args: List[Any]):
        inst = Instance(self)
        init = self.methods.get('init')
        if init:
            bound = BoundMethod(inst, init)
            bound(interp, args)
        elif args:
            raise RuntimeError(f"{self.name}.init expected 0 args, got {len(args)}")
        return inst

class Instance:
    def __init__(self, klass: 'Klass'):
        self.klass=klass
        self.fields: Dict[str, Any] = {}
    def get(self, name):
        if name in self.fields: return self.fields[name]
        m = self.klass.methods.get(name)
        if m: return BoundMethod(self, m)
        raise RuntimeError(f"Undefined property '{name}' on {self.klass.name}")
    def set(self, name, val):
        self.fields[name]=val

class BoundMethod:
    def __init__(self, instance: Instance, func: Function):
        self.instance=instance; self.func=func
    def __call__(self, interp, args: List[Any]):
        local = Env(self.func.closure)
        local.define('this', self.instance)
        if len(args) != len(self.func.params):
            raise RuntimeError(f"Method expects {len(self.func.params)} args, got {len(args)}")
        for p,a in zip(self.func.params,args): local.define(p,a)
        try:
            interp.exec_block(self.func.body, local)
        except ReturnSignal as rs:
            return rs.value
        return None

class Interpreter:
    def __init__(self, input_cb=None, print_cb=None):
        self.globals = Env()
        self.input_cb = input_cb or (lambda prompt="": input(prompt))
        self.print_cb  = print_cb  or (lambda *xs: print(*xs))
        self.install_builtins(self.globals)

    def install_builtins(self, env: Env):
        env.define('print', lambda *xs: self.print_cb(*xs))
        env.define('input', lambda p="": self.input_cb(p))
        env.define('len', lambda x: len(x))
        env.define('int', lambda x: int(x))
        env.define('float', lambda x: float(x))
        env.define('str', lambda x: str(x))
        env.define('type', lambda x: type(x).__name__)
        def run_os(cmd):
            try:
                out = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, universal_newlines=True)
                return out
            except subprocess.CalledProcessError as e:
                return e.output
        env.define('os', run_os)
        env.define('sqrt', lambda x: math.sqrt(x))
        env.define('pow', lambda a,b: pow(a,b))

    def truthy(self, v): return bool(v)

    def apply_bin(self, op, a, b):
        if op == '+': return a + b
        if op == '-': return a - b
        if op == '*': return a * b
        if op == '/': return a / b
        if op == '%': return a % b
        if op == '**': return a ** b
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
        if isinstance(node, Bool):   return node.value
        if isinstance(node, Null):   return None
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
                    if node.left.slice_end is not None:
                        raise RuntimeError("Slice assignment not supported")
                    idx = self.eval(node.left.index, env)
                    val = self.eval(node.right, env)
                    tgt[idx] = val; return val
                if isinstance(node.left, Member):
                    obj = self.eval(node.left.target, env)
                    if not isinstance(obj, Instance):
                        raise RuntimeError("Left of '.' is not an instance")
                    val = self.eval(node.right, env)
                    obj.set(node.left.name, val); return val
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
            if isinstance(node.callee, Var) and node.callee.name.startswith('__new__'):
                cname = node.callee.name[len('__new__'):]
                k: Klass = env.get(cname)
                if not isinstance(k, Klass): raise RuntimeError(f"'{cname}' is not a class")
                return k.instantiate(self, args)
            if isinstance(cal, Function): return cal(self, args)
            if isinstance(cal, BoundMethod): return cal(self, args)
            if callable(cal): return cal(*args)
            raise RuntimeError("Attempted to call a non-function")
        if isinstance(node, Index):
            tgt = self.eval(node.target, env)
            if node.slice_end is not None:
                start = self.eval(node.index, env) if node.index is not None else None
                end   = self.eval(node.slice_end, env) if not isinstance(node.slice_end, Null) else None
                return tgt[slice(start, end)]
            else:
                idx = self.eval(node.index, env)
                return tgt[idx]
        if isinstance(node, Member):
            obj = self.eval(node.target, env)
            if isinstance(obj, Instance):
                return obj.get(node.name)
            if isinstance(obj, dict):
                return obj.get(node.name)
            raise RuntimeError("Left of '.' is not an instance/dict")
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
            else:
                done=False
                for c,b in node.elifs:
                    if self.truthy(self.eval(c, env)):
                        self.exec_block(b, Env(env)); done=True; break
                if not done and node.else_b:
                    self.exec_block(node.else_b, Env(env))
            return None
        if isinstance(node, While):
            while self.truthy(self.eval(node.cond, env)):
                self.exec_block(node.body, Env(env))
            return None
        if isinstance(node, ForRange):
            a = self.eval(node.start, env)
            b = self.eval(node.end, env)
            step = 1 if b >= a else -1
            for v in range(a, b, step):
                loop = Env(env); loop.define(node.var, v)
                self.exec_block(node.body, loop)
            return None
        if isinstance(node, ForIn):
            it = self.eval(node.iterable, env)
            for v in it:
                loop = Env(env); loop.define(node.var, v)
                self.exec_block(node.body, loop)
            return None
        if isinstance(node, FuncDef):
            fn = Function(node.name, node.params, node.body, env)
            env.define(node.name, fn); return None
        if isinstance(node, ClassDef):
            methods: Dict[str, Function] = {}
            for mname, f in node.methods.items():
                methods[mname] = Function(f.name, f.params, f.body, env)
            env.define(node.name, Klass(node.name, methods)); return None
        if isinstance(node, Return):
            val = self.eval(node.expr, env) if node.expr is not None else None
            raise ReturnSignal(val)
        raise RuntimeError(f"Unknown AST node {node}")

    def exec_block(self, block: Block, env: Env):
        for st in block.statements:
            self.eval(st, env)

# ------------------------- Frontend helpers -------------------------

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

def run_file(path: str, interp: Optional[Interpreter]=None):
    with open(path, 'r', encoding='utf-8') as f:
        src = f.read()
    return run_source(src, interp)

# ------------------------- GUI (PyQt5) -------------------------

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog,
    QPlainTextEdit, QMessageBox, QSplitter, QInputDialog, QLabel
)
from PyQt5.QtGui import QFont, QColor, QTextCharFormat, QSyntaxHighlighter
from PyQt5.QtCore import Qt, QRegularExpression

class MinimaHighlighter(QSyntaxHighlighter):
    def __init__(self, parent):
        super().__init__(parent)
        self.rules = []

        def fmt(color, bold=False, italic=False):
            f = QTextCharFormat()
            f.setForeground(QColor(color))
            if bold: f.setFontWeight(QFont.Bold)
            if italic: f.setFontItalic(True)
            return f

        keyword = fmt("#73d0ff", True)
        builtin = fmt("#ffcc66", True)
        number  = fmt("#c4e88d")
        string  = fmt("#a6e22e")
        comment = fmt("#6c6c6c", italic=True)
        op      = fmt("#ff6e6e")

        kw = r"\b(let|var|func|return|if|elif|else|while|for|in|class|new|true|false|null)\b"
        bi = r"\b(print|input|len|int|float|str|type|os|sqrt|pow)\b"
        num = r"\b\d+(\.\d+)?\b"
        str_re = r"\".*?\"|'.*?'"
        line_comment = r"//[^\n]*|#[^\n]*"
        block_comment = r"/\*[\s\S]*?\*/"
        ops = r"==|!=|<=|>=|\+\+|--|\+|-|\*|/|%|\*\*|&&|\|\||!|=|<|>|\.\.|:"

        self.rules.append((QRegularExpression(kw), keyword))
        self.rules.append((QRegularExpression(bi), builtin))
        self.rules.append((QRegularExpression(num), number))
        self.rules.append((QRegularExpression(str_re), string))
        self.rules.append((QRegularExpression(ops), op))
        self.rules.append((QRegularExpression(line_comment), comment))
        self.block_comment_re = QRegularExpression(block_comment)
        self.block_comment_fmt = comment

    def highlightBlock(self, text):
        for regex, form in self.rules:
            it = regex.globalMatch(text)
            while it.hasNext():
                m = it.next()
                self.setFormat(m.capturedStart(), m.capturedLength(), form)
        # very simple multi-line block comment handling
        self.setCurrentBlockState(0)
        start = 0
        if self.previousBlockState() != 1:
            start = text.find("/*")
        else:
            start = 0
        while start >= 0:
            end = text.find("*/", start)
            if end == -1:
                self.setCurrentBlockState(1)
                length = len(text) - start
            else:
                length = end - start + 2
            self.setFormat(start, length, self.block_comment_fmt)
            if end == -1:
                break
            start = text.find("/*", end + 2)

class Console(QPlainTextEdit):
    def __init__(self):
        super().__init__()
        self.setReadOnly(True)
        self.setStyleSheet("QPlainTextEdit { background: #121212; color: #f0f0f0; border: 1px solid #333; }")
        self.setFont(QFont("Consolas", 11))

    def write(self, *args):
        s = " ".join(str(a) for a in args)
        self.appendPlainText(s)

    def clear_output(self):
        self.setPlainText("")

class Editor(QPlainTextEdit):
    def __init__(self):
        super().__init__()
        self.setStyleSheet("QPlainTextEdit { background: #0f111a; color: #d6deeb; border: 1px solid #333; }")
        self.setFont(QFont("Consolas", 12))
        self.highlighter = MinimaHighlighter(self.document())

DEFAULT_SAMPLE = """\
// Minima sample
let a = 5;
let b = [1,2,3, [4,5]];
print("a:", a);
print("b[2]:", b[2]);
print("b slice:", b[1:3]);

func add(x,y){ return x+y; }
print("add:", add(7,8));

class Person {
  func init(name){ this.name = name; }
  func greet(){ print("Hello, " + this.name); }
}

let p = new Person("Minima");
p.greet();

for i in 1..5 {
  print("range i:", i);
}

for x in [10,20,30] {
  print("for-in x:", x);
}

let name = input("Your name? ");
print("You typed:", name);

// Run an OS command (Windows example)
print(os("cmd /c echo Minima OK"));
"""

class MinimaApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Minima IDE — PyQt5")
        self.resize(1100, 700)
        self.apply_dark()

        self.editor = Editor()
        self.editor.setPlainText(DEFAULT_SAMPLE)

        self.console = Console()

        # Buttons
        self.btnRun = QPushButton("Run Editor")
        self.btnRunFile = QPushButton("Run File…")
        self.btnSave = QPushButton("Save")
        self.btnNew = QPushButton("New")
        self.btnClear = QPushButton("Clear Output")
        self.btnRunLine = QPushButton("Run Line")

        for b, color in [
            (self.btnRun, "#2e7d32"),
            (self.btnRunFile, "#1565c0"),
            (self.btnSave, "#6d4c41"),
            (self.btnNew, "#00838f"),
            (self.btnClear, "#b71c1c"),
            (self.btnRunLine, "#7b1fa2"),
        ]:
            b.setStyleSheet(f"QPushButton {{ background: {color}; color: white; padding: 8px 12px; border-radius: 4px; }} QPushButton:hover {{ filter: brightness(120%); }}")

        topbar = QHBoxLayout()
        topbar.addWidget(self.btnRun)
        topbar.addWidget(self.btnRunFile)
        topbar.addWidget(self.btnSave)
        topbar.addWidget(self.btnNew)
        topbar.addWidget(self.btnRunLine)
        topbar.addStretch(1)
        topbar.addWidget(self.btnClear)

        split = QSplitter(Qt.Vertical)
        split.addWidget(self.editor)
        split.addWidget(self.console)
        split.setSizes([450, 250])

        layout = QVBoxLayout(self)
        layout.addLayout(topbar)
        layout.addWidget(split)

        # Interpreter bound to GUI I/O
        self.interp = Interpreter(
            input_cb=self.gui_input,
            print_cb=self.console.write
        )

        # Wire buttons
        self.btnRun.clicked.connect(self.run_editor)
        self.btnRunFile.clicked.connect(self.run_file)
        self.btnSave.clicked.connect(self.save_file)
        self.btnNew.clicked.connect(self.new_file)
        self.btnClear.clicked.connect(self.console.clear_output)
        self.btnRunLine.clicked.connect(self.run_current_line)

    def apply_dark(self):
        self.setStyleSheet("""
        QWidget { background: #0b0e14; color: #e6edf3; }
        QPushButton { font-weight: 600; }
        QSplitter::handle { background: #222; }
        """)

    def gui_input(self, prompt=""):
        text, ok = QInputDialog.getText(self, "Minima input()", prompt)
        return text if ok else ""

    def run_editor(self):
        code = self.editor.toPlainText()
        try:
            run_source(code, self.interp)
        except Exception as e:
            QMessageBox.critical(self, "Runtime Error", str(e))

    def run_current_line(self):
        cursor = self.editor.textCursor()
        cursor.select(cursor.LineUnderCursor)
        code = cursor.selectedText()
        if not code.strip():
            return
        # Ensure semicolon for single-line
        if not code.strip().endswith(';'):
            code = code.rstrip() + ';'
        try:
            run_source(code, self.interp)
        except Exception as e:
            QMessageBox.critical(self, "Runtime Error", str(e))

    def run_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Run .minima file", "", "Minima (*.minima);;All files (*.*)")
        if not path: return
        try:
            run_file(path, self.interp)
        except Exception as e:
            QMessageBox.critical(self, "Runtime Error", f"{e}\n\nFile: {path}")

    def save_file(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save code", "", "Minima (*.minima);;All files (*.*)")
        if not path: return
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(self.editor.toPlainText())
            QMessageBox.information(self, "Saved", f"Saved to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))

    def new_file(self):
        self.editor.setPlainText("// New Minima file\n")

# ------------------------- Main -------------------------

def main():
    app = QApplication(sys.argv)
    w = MinimaApp()
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

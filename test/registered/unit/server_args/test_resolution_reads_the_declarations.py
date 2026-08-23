"""Resolution reads its own decisions, not the record's fields.

`declare_resolution` records a decision in the declaration stash and writes
nothing. The fields keep what the caller passed, so a resolver that reads a
field another resolver may have decided reads the raw input -- silently, and
only on the configurations where that other resolver fires. The whole pipeline
therefore reads through `resolving_view` (or `ServerArgs._resolved()`, which is
the same view spelled as the record's own member), and this pins that there is
nothing left reading a field directly.

Subjects: every function in `arg_groups/` that takes a config, every
`ServerArgs` handler the dispatcher reaches, and every member of `ServerArgs` /
`PortArgs` -- the members are reached from the hooks and from business code
rather than from the dispatcher, so the handler walk cannot see them, and a
member that recomputes from a raw field decides from what was typed. All three
are derived -- a new hook file, a new handler or a new member is covered the
moment it is written. Readers *outside* those
two -- the platform defaults, `ModelConfig`, the spec-algo hook -- are reached by
resolution too and have moved to the view as well, but enumerating them needs
the call-graph derivation `test_resolution_reads_no_bag` owns; this file pins
the two scopes it can derive exactly.
"""

import ast
import dataclasses
import pathlib

import sglang
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

_SRT = pathlib.Path(sglang.__file__).resolve().parent / "srt"
_FIELDS = frozenset(field.name for field in dataclasses.fields(ServerArgs))

# Names a config travels under. `args` is included because the platform hooks
# use it; a false positive would be a function taking an argparse Namespace and
# reading an attribute that happens to be a ServerArgs field name, which the
# allowlist below would then have to carry.
_HOLDER_NAMES = frozenset({"server_args", "sa", "args"})


def _holders(fn):
    names = {
        arg.arg
        for arg in list(fn.args.posonlyargs)
        + list(fn.args.args)
        + list(fn.args.kwonlyargs)
        if arg.arg in _HOLDER_NAMES
    }
    for arg in (
        list(fn.args.posonlyargs) + list(fn.args.args) + list(fn.args.kwonlyargs)
    ):
        annotation = arg.annotation
        text = (
            annotation.value
            if isinstance(annotation, ast.Constant)
            else (
                annotation.id
                if isinstance(annotation, ast.Name)
                else annotation.attr if isinstance(annotation, ast.Attribute) else None
            )
        )
        if text == "ServerArgs":
            names.add(arg.arg)
    return names


def _field_reads(fn, holders):
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.Attribute)
            and node.attr in _FIELDS
            and isinstance(node.value, ast.Name)
            and node.value.id in holders
            and isinstance(node.ctx, ast.Load)
        ):
            yield node.lineno, node.attr


def _resolution_handlers():
    """The `ServerArgs` methods the dispatcher reaches, transitively."""
    tree = ast.parse((_SRT / "server_args.py").read_text(encoding="utf-8-sig"))
    cls = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "ServerArgs"
    )
    methods = {
        node.name: node
        for node in cls.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "_run_resolution_pipeline" in methods, "the dispatcher was renamed"
    seen, stack = set(), ["_run_resolution_pipeline"]
    while stack:
        name = stack.pop()
        if name in seen or name not in methods:
            continue
        seen.add(name)
        for node in ast.walk(methods[name]):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "self"
            ):
                stack.append(node.func.attr)
    return {name: methods[name] for name in seen}


_DECLARERS = frozenset(
    {
        "_declare",
        "declare_resolution",
        "declare_late_resolution",
        "declare_direct_writes",
    }
)


def _declared_fields():
    """The fields resolution decides, read off the declaration calls.

    Two shapes reach the stash: a literal keyword (`declare_resolution(sa, src,
    page_size=64)`) and a dict splatted into the call (`**overrides`), which is
    how the model-override and post-process passes carry theirs. Missing the
    second shape would leave `dtype`, `fp8_gemm_runner_backend` and eleven more
    outside the subject set.
    """
    fields, splatted = set(), set()
    sources = [_SRT / "server_args.py"] + sorted((_SRT / "arg_groups").glob("*.py"))
    trees = {}
    for path in sources:
        trees[path] = ast.parse(path.read_text(encoding="utf-8-sig"))
        for node in ast.walk(trees[path]):
            if not isinstance(node, ast.Call):
                continue
            name = (
                node.func.id
                if isinstance(node.func, ast.Name)
                else getattr(node.func, "attr", None)
            )
            if name not in _DECLARERS:
                continue
            for keyword in node.keywords:
                if keyword.arg:
                    fields.add(keyword.arg)
                elif isinstance(keyword.value, ast.Name):
                    splatted.add(keyword.value.id)
                elif isinstance(keyword.value, ast.Dict):
                    fields.update(
                        key.value
                        for key in keyword.value.keys
                        if isinstance(key, ast.Constant) and isinstance(key.value, str)
                    )
    for tree in trees.values():
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Assign)
                and isinstance(node.targets[0], ast.Subscript)
                and isinstance(node.targets[0].value, ast.Name)
                and node.targets[0].value.id in splatted
                and isinstance(node.targets[0].slice, ast.Constant)
                and isinstance(node.targets[0].slice.value, str)
            ):
                fields.add(node.targets[0].slice.value)
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("update", "setdefault")
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in splatted
            ):
                for arg in node.args:
                    if isinstance(arg, ast.Dict):
                        fields.update(
                            key.value
                            for key in arg.keys
                            if isinstance(key, ast.Constant)
                            and isinstance(key.value, str)
                        )
                    elif isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        fields.add(arg.value)
    return frozenset(fields & _FIELDS)


def _record_members():
    """Every member of `ServerArgs` / `PortArgs`, by class and name."""
    tree = ast.parse((_SRT / "server_args.py").read_text(encoding="utf-8-sig"))
    members = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name in ("ServerArgs", "PortArgs"):
            for member in node.body:
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    members[f"{node.name}.{member.name}"] = member
    return members


class TestResolutionReadsTheDeclarations(CustomTestCase):
    def test_no_hook_reads_a_field_off_the_record(self):
        offenders = []
        files = sorted((_SRT / "arg_groups").glob("*.py"))
        self.assertGreater(len(files), 5, "the hook scan found almost nothing")
        for path in files:
            rel = f"arg_groups/{path.name}"
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
            for fn in ast.walk(tree):
                if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                holders = _holders(fn)
                if not holders:
                    continue
                for lineno, field in _field_reads(fn, holders):
                    offenders.append(f"{rel}:{lineno} {fn.name} reads .{field}")
        self.assertEqual(
            offenders,
            [],
            "a resolution hook reads a field off the record; the field holds the "
            "raw input, so this decides from what was typed rather than from "
            "what resolution decided. Read `resolving_view(server_args)`:\n  "
            + "\n  ".join(offenders),
        )

    def test_no_handler_reads_a_field_off_self(self):
        handlers = _resolution_handlers()
        self.assertGreater(
            len(handlers), 50, f"only {len(handlers)} handlers were reached"
        )
        offenders = []
        for name, fn in sorted(handlers.items()):
            for lineno, field in _field_reads(fn, {"self"}):
                offenders.append(f"server_args.py:{lineno} {name} reads self.{field}")
        self.assertEqual(
            offenders,
            [],
            "a resolution handler reads its own field; the field holds the raw "
            "input. Bind `cfg = resolving_view(self)` and read that:\n  "
            + "\n  ".join(offenders),
        )

    def test_no_member_recomputes_from_a_raw_field(self):
        decided = _declared_fields()
        self.assertGreater(
            len(decided), 100, f"the declaration set derived only {len(decided)} fields"
        )
        members = _record_members()
        self.assertGreater(len(members), 100, f"only {len(members)} members were found")
        offenders = []
        for name, fn in sorted(members.items()):
            holders = _holders(fn) | {"self"}
            for lineno, field in _field_reads(fn, holders):
                if field in decided:
                    offenders.append(f"server_args.py:{lineno} {name} reads .{field}")
        self.assertEqual(
            offenders,
            [],
            "a record member recomputes from a field resolution decides; the "
            "field holds the raw input, so the member answers for what was "
            "typed. Bind `cfg = resolving_view(self)` and read that:\n  "
            + "\n  ".join(offenders),
        )


if __name__ == "__main__":
    import unittest

    unittest.main()

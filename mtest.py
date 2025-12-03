import streamlit as st
import ast
import sys
import types
import importlib.util
import time
import traceback
import tracemalloc
from typing import Any, Dict, List, Optional, Tuple

st.set_page_config(page_title="Function Tester", layout="wide")

class FuncInfo:
    def __init__(self, name: str, node: ast.AST, signature_text: str, params: List[Dict[str, Any]], assigned_vars: List[str], lineno: int, end_lineno: Optional[int], class_name: Optional[str], is_method: bool):
        self.name = name
        self.node = node
        self.signature_text = signature_text
        self.params = params
        self.assigned_vars = assigned_vars
        self.lineno = lineno
        self.end_lineno = end_lineno
        self.class_name = class_name
        self.is_method = is_method

def _ann_to_text(source: str, ann: Optional[ast.AST]) -> Optional[str]:
    if ann is None:
        return None
    try:
        if hasattr(ast, "unparse"):
            return ast.unparse(ann)
    except Exception:
        pass
    try:
        seg = ast.get_source_segment(source, ann)
        return seg
    except Exception:
        return None

def parse_functions(source: str) -> List[FuncInfo]:
    tree = ast.parse(source)
    funcs: List[FuncInfo] = []

    def build_info(node: ast.AST, class_name: Optional[str]) -> FuncInfo:
        args = node.args  # type: ignore[attr-defined]
        parts: List[str] = []
        params: List[Dict[str, Any]] = []
        posonly = getattr(args, "posonlyargs", [])
        defaults = list(args.defaults)
        kw_defaults = list(args.kw_defaults)
        total_pos = len(posonly) + len(args.args)
        num_defaults = len(defaults)
        pos_default_start = total_pos - num_defaults
        i = 0
        for a in posonly:
            d = None
            if i >= pos_default_start and defaults:
                d = defaults.pop(0)
            ann_text = _ann_to_text(source, a.annotation)
            name = a.arg
            if ann_text:
                text = f"{name}: {ann_text}"
            else:
                text = name
            if d is not None:
                text += " = ..."
            parts.append(text)
            params.append({"kind": "posonly", "name": name, "annotation": ann_text, "has_default": d is not None})
            i += 1
        if posonly:
            parts.append("/")
        for a in args.args:
            d = None
            if i >= pos_default_start and defaults:
                d = defaults.pop(0)
            ann_text = _ann_to_text(source, a.annotation)
            name = a.arg
            if ann_text:
                text = f"{name}: {ann_text}"
            else:
                text = name
            if d is not None:
                text += " = ..."
            parts.append(text)
            params.append({"kind": "pos_or_kw", "name": name, "annotation": ann_text, "has_default": d is not None})
            i += 1
        if args.vararg is not None:
            ann_text = _ann_to_text(source, args.vararg.annotation)
            name = args.vararg.arg
            if ann_text:
                parts.append(f"*{name}: {ann_text}")
            else:
                parts.append(f"*{name}")
            params.append({"kind": "vararg", "name": name, "annotation": ann_text, "has_default": False})
        if args.kwonlyargs:
            if args.vararg is None:
                parts.append("*")
            for idx, a in enumerate(args.kwonlyargs):
                d = kw_defaults[idx] if idx < len(kw_defaults) else None
                ann_text = _ann_to_text(source, a.annotation)
                name = a.arg
                if ann_text:
                    text = f"{name}: {ann_text}"
                else:
                    text = name
                if d is not None:
                    text += " = ..."
                parts.append(text)
                params.append({"kind": "kwonly", "name": name, "annotation": ann_text, "has_default": d is not None})
        if args.kwarg is not None:
            ann_text = _ann_to_text(source, args.kwarg.annotation)
            name = args.kwarg.arg
            if ann_text:
                parts.append(f"**{name}: {ann_text}")
            else:
                parts.append(f"**{name}")
            params.append({"kind": "kwarg", "name": name, "annotation": ann_text, "has_default": False})
        ret_ann = _ann_to_text(source, getattr(node, "returns", None))
        sig_text = f"def {node.name}(" + ", ".join(parts) + ")"  # type: ignore[attr-defined]
        if ret_ann:
            sig_text += f" -> {ret_ann}"
        assigned: List[str] = []
        for inner in ast.walk(node):
            if isinstance(inner, ast.Assign):
                for t in inner.targets:
                    if isinstance(t, ast.Name):
                        assigned.append(t.id)
            elif isinstance(inner, ast.AnnAssign):
                t = inner.target
                if isinstance(t, ast.Name):
                    assigned.append(t.id)
            elif isinstance(inner, ast.AugAssign):
                t = inner.target
                if isinstance(t, ast.Name):
                    assigned.append(t.id)
        return FuncInfo(node.name, node, sig_text, params, sorted(set(assigned)), node.lineno, getattr(node, "end_lineno", None), class_name, class_name is not None)  # type: ignore[attr-defined]

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            funcs.append(build_info(node, None))
        elif isinstance(node, ast.ClassDef):
            for inner in node.body:
                if isinstance(inner, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    funcs.append(build_info(inner, node.name))
    funcs.sort(key=lambda f: f.lineno)
    return funcs

def extract_imports(source: str) -> List[str]:
    tree = ast.parse(source)
    mods: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mods.append(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                mods.append(node.module.split(".")[0])
    seen = []
    for m in mods:
        if m not in seen:
            seen.append(m)
    return seen

def check_modules_availability(mods: List[str]) -> Tuple[List[str], List[str]]:
    available: List[str] = []
    missing: List[str] = []
    for m in mods:
        try:
            spec = importlib.util.find_spec(m)
            if spec is not None:
                available.append(m)
            else:
                missing.append(m)
        except Exception:
            missing.append(m)
    return available, missing

def create_module_from_source(source: str, filename: str = "uploaded_module.py") -> types.ModuleType:
    module = types.ModuleType("uploaded_module")
    module.__file__ = filename
    module.__package__ = None
    code = compile(source, filename, "exec")
    exec(code, module.__dict__)
    return module

def parse_literal(text: str) -> Any:
    import ast as _ast
    try:
        return _ast.literal_eval(text)
    except Exception:
        return text

def cast_input(val: Any, ann: Optional[str]) -> Any:
    if ann is None:
        return val
    try:
        if ann in ("int", "builtins.int"):
            return int(val)
        if ann in ("float", "builtins.float"):
            return float(val)
        if ann in ("bool", "builtins.bool"):
            if isinstance(val, bool):
                return val
            if isinstance(val, str):
                v = val.strip().lower()
                if v in ("true", "1", "yes", "y", "on"):
                    return True
                if v in ("false", "0", "no", "n", "off"):
                    return False
            return bool(val)
        if ann in ("str", "builtins.str"):
            return str(val)
        if ann in ("list", "builtins.list"):
            if isinstance(val, list):
                return val
            return list(parse_literal(val))
        if ann in ("dict", "builtins.dict"):
            if isinstance(val, dict):
                return val
            return dict(parse_literal(val))
        if ann in ("tuple", "builtins.tuple"):
            if isinstance(val, tuple):
                return val
            pv = parse_literal(val)
            return tuple(pv if isinstance(pv, (list, tuple)) else [pv])
        if ann in ("set", "builtins.set"):
            if isinstance(val, set):
                return val
            pv = parse_literal(val)
            return set(pv if isinstance(pv, (list, tuple, set)) else [pv])
    except Exception:
        return val
    return val

def run_with_trace(func, args: List[Any], kwargs: Dict[str, Any]) -> Tuple[Any, List[Dict[str, Any]], Optional[BaseException], Optional[str]]:
    timeline: List[Dict[str, Any]] = []
    target_code = func.__code__
    def tracer(frame, event, arg):
        if frame.f_code is not target_code:
            return tracer
        if event in ("line", "return"):
            try:
                snap = dict(frame.f_locals)
            except Exception:
                snap = {}
            timeline.append({"event": event, "line": frame.f_lineno, "locals": snap})
        if event == "exception":
            try:
                exc, val, tb = arg
                snap = dict(frame.f_locals)
            except Exception:
                snap = {}
            timeline.append({"event": event, "line": frame.f_lineno, "locals": snap})
        return tracer
    result = None
    err: Optional[BaseException] = None
    tb_str: Optional[str] = None
    old_trace = sys.gettrace()
    try:
        sys.settrace(tracer)
        result = func(*args, **kwargs)
    except BaseException as e:
        err = e
        tb_str = traceback.format_exc()
    finally:
        sys.settrace(old_trace)
    return result, timeline, err, tb_str

st.title("Python Function Tester")

uploaded = st.file_uploader("Upload a Python file", type=["py"])

if uploaded is None:
    st.info("Upload a .py file to begin.")
    st.stop()

source_bytes = uploaded.read()
try:
    source_text = source_bytes.decode("utf-8")
except Exception:
    source_text = source_bytes.decode("latin-1", errors="ignore")

left, right = st.columns([1, 1])
with left:
    st.subheader("File Contents")
    st.code(source_text, language="python")

funcs = parse_functions(source_text)
imports = extract_imports(source_text)
avail, missing = check_modules_availability(imports)

with right:
    st.subheader("Imports")
    if imports:
        st.write({"available": avail, "missing": missing})
    else:
        st.write("No imports detected")
    st.subheader("Functions")
    if not funcs:
        st.warning("No functions or methods found in the file.")
    viewable_funcs = [f for f in funcs if not (f.is_method and f.name == "__init__")]
    if not viewable_funcs:
        st.warning("No runnable functions found (only __init__ detected).")
        st.stop()
    options_idx = list(range(len(viewable_funcs)))
    def _fmt(i: int) -> str:
        f = viewable_funcs[i]
        kind = "method" if f.is_method else "function"
        prefix = f"{f.class_name}." if f.is_method else ""
        return f"{prefix}{f.name} ({kind})"
    selected_idx = st.selectbox("Select a function", options=options_idx, format_func=_fmt)

selected = viewable_funcs[selected_idx] if 'viewable_funcs' in locals() and viewable_funcs else None

if selected is None:
    st.stop()

st.markdown(f"### Signature\n``{selected.signature_text}``")
st.caption(f"Lines {selected.lineno} - {selected.end_lineno or '?'}")
st.markdown("#### Variables Assigned Inside Function")
st.write(selected.assigned_vars or "None")

module = None
load_error = None
try:
    module = create_module_from_source(source_text, filename=uploaded.name)
except Exception as e:
    load_error = traceback.format_exc()

if load_error is not None:
    st.error("Failed to load module")
    st.exception(load_error)
    st.stop()

if not selected.is_method:
    if not hasattr(module, selected.name):
        st.error("Selected function not found at runtime.")
        st.stop()
else:
    if not hasattr(module, selected.class_name):
        st.error("Selected class not found at runtime.")
        st.stop()
    class_obj = getattr(module, selected.class_name)
    st.markdown("### Class Instance")
    inst_key = f"__instance__::{uploaded.name}::{selected.class_name}"
    instance_ready = inst_key in st.session_state
    if instance_ready:
        st.success(f"Instance of {selected.class_name} is ready.")
        if st.button("Clear Instance"):
            try:
                del st.session_state[inst_key]
            except Exception:
                pass
            instance_ready = False
    if not instance_ready:
        init_info = None
        for f in funcs:
            if f.is_method and f.class_name == selected.class_name and f.name == "__init__":
                init_info = f
                break
        with st.form(f"init_form_{selected.class_name}"):
            init_widgets: Dict[str, Any] = {}
            init_pos: List[str] = []
            init_kwonly: List[str] = []
            init_vararg: Optional[str] = None
            init_varkw: Optional[str] = None
            init_params = init_info.params if init_info is not None else []
            if init_params and init_params[0].get("name") == "self":
                init_params = init_params[1:]
            st.caption("Provide constructor (__init__) arguments")
            for p in init_params:
                kind = p["kind"]
                name = p["name"]
                ann = p["annotation"]
                label = f"{name} ({ann})" if ann else name
                if kind in ("posonly", "pos_or_kw"):
                    init_pos.append(name)
                    if ann in ("int", "builtins.int"):
                        init_widgets[name] = st.number_input(label, step=1, format="%d", key=f"init_{name}")
                    elif ann in ("float", "builtins.float"):
                        init_widgets[name] = st.number_input(label, value=0.0, key=f"init_{name}")
                    elif ann in ("bool", "builtins.bool"):
                        init_widgets[name] = st.checkbox(label, value=False, key=f"init_{name}")
                    elif ann in ("str", "builtins.str"):
                        init_widgets[name] = st.text_input(label, value="", key=f"init_{name}")
                    elif ann in ("list", "dict", "tuple", "set", "builtins.list", "builtins.dict", "builtins.tuple", "builtins.set"):
                        init_widgets[name] = st.text_area(label, placeholder=f"Enter {ann} literal", key=f"init_{name}")
                    else:
                        init_widgets[name] = st.text_input(label, placeholder="Python literal or string", key=f"init_{name}")
                elif kind == "vararg":
                    init_vararg = name
                    init_widgets[name] = st.text_area(f"*{name}", placeholder="Comma-separated or Python list/tuple literal", key=f"init_{name}")
                elif kind == "kwonly":
                    init_kwonly.append(name)
                    if ann in ("int", "builtins.int"):
                        init_widgets[name] = st.number_input(label, step=1, format="%d", key=f"init_kw_{name}")
                    elif ann in ("float", "builtins.float"):
                        init_widgets[name] = st.number_input(label, value=0.0, key=f"init_kw_{name}")
                    elif ann in ("bool", "builtins.bool"):
                        init_widgets[name] = st.checkbox(label, value=False, key=f"init_kw_{name}")
                    elif ann in ("str", "builtins.str"):
                        init_widgets[name] = st.text_input(label, value="", key=f"init_kw_{name}")
                    elif ann in ("list", "dict", "tuple", "set", "builtins.list", "builtins.dict", "builtins.tuple", "builtins.set"):
                        init_widgets[name] = st.text_area(label, placeholder=f"Enter {ann} literal", key=f"init_kw_{name}")
                    else:
                        init_widgets[name] = st.text_input(label, placeholder="Python literal or string", key=f"init_kw_{name}")
                elif kind == "kwarg":
                    init_varkw = name
                    init_widgets[name] = st.text_area(f"**{name}", placeholder="Python dict literal for extra keyword args", key=f"init_{name}")
            create_inst = st.form_submit_button("Create Instance")
        if create_inst:
            try:
                inst_args: List[Any] = []
                inst_kwargs: Dict[str, Any] = {}
                for n in init_pos:
                    val = init_widgets[n]
                    ann = None
                    for p in init_params:
                        if p["name"] == n:
                            ann = p["annotation"]
                            break
                    casted = cast_input(val, ann)
                    if isinstance(val, str) and ann not in ("str", "builtins.str") and val.strip() != "":
                        casted = parse_literal(val)
                    if val == "" and ann not in ("str", "builtins.str"):
                        pass
                    else:
                        inst_args.append(casted)
                if init_vararg is not None:
                    raw = init_widgets.get(init_vararg, "")
                    if isinstance(raw, str) and raw.strip():
                        try:
                            pv = parse_literal(raw)
                            if isinstance(pv, (list, tuple)):
                                inst_args.extend(list(pv))
                            else:
                                inst_args.append(pv)
                        except Exception:
                            parts = [x.strip() for x in raw.split(",") if x.strip()]
                            inst_args.extend(parts)
                for n in init_kwonly:
                    val = init_widgets[n]
                    ann = None
                    for p in init_params:
                        if p["name"] == n:
                            ann = p["annotation"]
                            break
                    casted = cast_input(val, ann)
                    if isinstance(val, str) and ann not in ("str", "builtins.str") and val.strip() != "":
                        casted = parse_literal(val)
                    if val == "" and ann not in ("str", "builtins.str"):
                        pass
                    else:
                        inst_kwargs[n] = casted
                if init_varkw is not None:
                    raw = init_widgets.get(init_varkw, "")
                    if isinstance(raw, str) and raw.strip():
                        try:
                            pv = parse_literal(raw)
                            if isinstance(pv, dict):
                                inst_kwargs.update(pv)
                        except Exception:
                            pass
                instance = class_obj(*inst_args, **inst_kwargs)
                st.session_state[inst_key] = instance
                st.success("Instance created.")
                instance_ready = True
            except BaseException as e:
                st.error(f"Failed to create instance: {type(e).__name__}: {e}")

st.markdown("### Provide Arguments")
with st.form("args_form"):
    widgets: Dict[str, Any] = {}
    pos_param_order: List[str] = []
    kwonly_param_order: List[str] = []
    vararg_name: Optional[str] = None
    varkw_name: Optional[str] = None
    params_for_ui = list(selected.params)
    if selected.is_method and params_for_ui and params_for_ui[0].get("name") == "self":
        params_for_ui = params_for_ui[1:]
    for p in params_for_ui:
        kind = p["kind"]
        name = p["name"]
        ann = p["annotation"]
        label = f"{name} ({ann})" if ann else name
        if kind in ("posonly", "pos_or_kw"):
            pos_param_order.append(name)
            if ann in ("int", "builtins.int"):
                widgets[name] = st.number_input(label, step=1, format="%d")
            elif ann in ("float", "builtins.float"):
                widgets[name] = st.number_input(label, value=0.0)
            elif ann in ("bool", "builtins.bool"):
                widgets[name] = st.checkbox(label, value=False)
            elif ann in ("str", "builtins.str"):
                widgets[name] = st.text_input(label, value="")
            elif ann in ("list", "dict", "tuple", "set", "builtins.list", "builtins.dict", "builtins.tuple", "builtins.set"):
                widgets[name] = st.text_area(label, placeholder=f"Enter {ann} literal, e.g., [1,2] or {'{'}'a':1{'}'}")
            else:
                widgets[name] = st.text_input(label, placeholder="Python literal or string")
        elif kind == "vararg":
            vararg_name = name
            widgets[name] = st.text_area(f"*{name}", placeholder="Comma-separated or Python list/tuple literal")
        elif kind == "kwonly":
            kwonly_param_order.append(name)
            if ann in ("int", "builtins.int"):
                widgets[name] = st.number_input(label, step=1, format="%d", key=f"kw_{name}")
            elif ann in ("float", "builtins.float"):
                widgets[name] = st.number_input(label, value=0.0, key=f"kw_{name}")
            elif ann in ("bool", "builtins.bool"):
                widgets[name] = st.checkbox(label, value=False, key=f"kw_{name}")
            elif ann in ("str", "builtins.str"):
                widgets[name] = st.text_input(label, value="", key=f"kw_{name}")
            elif ann in ("list", "dict", "tuple", "set", "builtins.list", "builtins.dict", "builtins.tuple", "builtins.set"):
                widgets[name] = st.text_area(label, placeholder=f"Enter {ann} literal", key=f"kw_{name}")
            else:
                widgets[name] = st.text_input(label, placeholder="Python literal or string", key=f"kw_{name}")
        elif kind == "kwarg":
            varkw_name = name
            widgets[name] = st.text_area(f"**{name}", placeholder="Python dict literal for extra keyword args")
    submit = st.form_submit_button("Run Function")

result = None
timeline: List[Dict[str, Any]] = []
err: Optional[BaseException] = None
tb_str: Optional[str] = None
exec_time = None
mem_peak = None

if submit:
    call_args: List[Any] = []
    call_kwargs: Dict[str, Any] = {}
    try:
        for n in pos_param_order:
            val = widgets[n]
            ann = None
            for p in params_for_ui:
                if p["name"] == n:
                    ann = p["annotation"]
                    break
            casted = cast_input(val, ann)
            if isinstance(val, str) and ann not in ("str", "builtins.str") and val.strip() != "":
                casted = parse_literal(val)
            if val == "" and ann not in ("str", "builtins.str"):
                pass
            else:
                call_args.append(casted)
        if vararg_name is not None:
            raw = widgets.get(vararg_name, "")
            if isinstance(raw, str) and raw.strip():
                try:
                    pv = parse_literal(raw)
                    if isinstance(pv, (list, tuple)):
                        call_args.extend(list(pv))
                    else:
                        call_args.append(pv)
                except Exception:
                    parts = [x.strip() for x in raw.split(",") if x.strip()]
                    call_args.extend(parts)
        for n in kwonly_param_order:
            val = widgets[n]
            ann = None
            for p in params_for_ui:
                if p["name"] == n:
                    ann = p["annotation"]
                    break
            casted = cast_input(val, ann)
            if isinstance(val, str) and ann not in ("str", "builtins.str") and val.strip() != "":
                casted = parse_literal(val)
            if val == "" and ann not in ("str", "builtins.str"):
                pass
            else:
                call_kwargs[n] = casted
        if varkw_name is not None:
            raw = widgets.get(varkw_name, "")
            if isinstance(raw, str) and raw.strip():
                try:
                    pv = parse_literal(raw)
                    if isinstance(pv, dict):
                        call_kwargs.update(pv)
                except Exception:
                    pass
        if selected.is_method:
            inst_key = f"__instance__::{uploaded.name}::{selected.class_name}"
            if inst_key not in st.session_state:
                raise RuntimeError("No class instance available. Create an instance via __init__ first.")
            bound = getattr(st.session_state[inst_key], selected.name)
            target_callable = bound
        else:
            target_callable = getattr(module, selected.name)
        tracemalloc.start()
        t0 = time.perf_counter()
        result, timeline, err, tb_str = run_with_trace(target_callable, call_args, call_kwargs)
        t1 = time.perf_counter()
        current, peak = tracemalloc.get_traced_memory()
        mem_peak = peak
        tracemalloc.stop()
        exec_time = t1 - t0
    except BaseException as e:
        err = e
        tb_str = traceback.format_exc()
        try:
            tracemalloc.stop()
        except Exception:
            pass

st.markdown("### Results")
if submit:
    if err is not None:
        st.error(f"Error: {type(err).__name__}: {err}")
        if tb_str:
            st.exception(tb_str)
    else:
        col1, col2 = st.columns([1,1])
        with col1:
            st.subheader("Return Value")
            st.write(result)
            st.caption(f"Type: {type(result).__name__}")
        with col2:
            st.subheader("Performance")
            if exec_time is not None:
                st.write({"elapsed_seconds": exec_time})
            if mem_peak is not None:
                st.write({"peak_memory_bytes": mem_peak})

    if timeline:
        st.markdown("### Trace Timeline")
        st.caption("Locals snapshot by line inside the function")
        lines = [f["line"] for f in timeline]
        idx = st.slider("Step", 1, len(timeline), len(timeline))
        snap = timeline[idx-1]
        st.write({"event": snap["event"], "line": snap["line"]})
        st.write(snap["locals"]) 
        last_locals = timeline[-1]["locals"]
        tracked = {k: last_locals.get(k, None) for k in selected.assigned_vars}
        st.markdown("#### Final Tracked Variables")
        st.write(tracked)

st.markdown("### Optional: Quick Test")
with st.expander("Compare result to expected"):
    expected_text = st.text_input("Expected (Python literal)")
    if submit and err is None and expected_text.strip():
        try:
            expected_val = parse_literal(expected_text)
            ok = (result == expected_val)
            if ok:
                st.success("Assertion passed: result == expected")
            else:
                st.warning("Assertion failed")
                st.write({"result": result, "expected": expected_val})
        except Exception as e:
            st.error(f"Failed to parse expected: {e}")

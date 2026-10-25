"""Standalone ONNX/QNN runner: copy this one file alongside ONNX/BIN assets.

Dependencies: numpy, a platform-compatible onnxruntime with QNN EP, QAIRT.
Random mode tests each graph independently; it does not run a GR00T policy.
"""

import argparse
import importlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


def package_files(package):
    """Read the installed Ubuntu package inventory; never install anything."""
    try:
        result = subprocess.run(
            ["dpkg", "-L", package], capture_output=True, text=True,
            check=False, timeout=10,
        )
        return [Path(line) for line in result.stdout.splitlines()] if result.returncode == 0 else []
    except (OSError, subprocess.TimeoutExpired):
        return []


def find_backend(explicit=None):
    if explicit is not None:
        if not explicit.is_file():
            raise FileNotFoundError(f"QNN backend does not exist: {explicit}")
        return explicit.resolve()
    # Prefer the board's package over SDKs for other host/target platforms.
    candidates = [p for p in package_files("qairt-libs")
                  if p.name == "libQnnHtp.so" and p.is_file()]
    if not candidates:
        for folder in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep):
            if folder:
                path = Path(folder) / "libQnnHtp.so"
                if path.is_file():
                    candidates.append(path)
        for folder in (Path("/usr/lib"), Path("/usr/local/lib"), Path("/opt/qairt")):
            if folder.exists():
                candidates.extend(folder.rglob("libQnnHtp.so"))
    candidates = sorted({p.resolve() for p in candidates if p.is_file()})
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            "libQnnHtp.so not found. On stock VENTUNO Q Ubuntu, install the board "
            "runtime: sudo apt update && sudo apt install qairt-libs qairt-dsp-binaries. "
            "For a custom SDK, specify --backend /path/to/libQnnHtp.so."
        )
    raise ValueError(f"Multiple QNN backends found: {candidates}; choose one with --backend")


def load_qnn_plugin(plugin_dir=None):
    """Import the active environment's QNN 2.x plugin, or an explicit package."""
    if plugin_dir is not None:
        plugin_dir = plugin_dir.expanduser().resolve()
        if plugin_dir.name != "onnxruntime_qnn" or not (plugin_dir / "__init__.py").is_file():
            raise ValueError("--qnn-plugin-dir must point to the onnxruntime_qnn package directory")
        sys.path.insert(0, str(plugin_dir.parent))
    try:
        plugin = importlib.import_module("onnxruntime_qnn")
    except ModuleNotFoundError as error:
        if error.name == "onnxruntime_qnn" and plugin_dir is None:
            return None  # The older bundled ORT/QNN wheel has no plugin module.
        raise
    for name in ("get_library_path", "get_qnn_htp_path"):
        if not callable(getattr(plugin, name, None)):
            raise RuntimeError(f"QNN package lacks {name}; use a QNN 2.x plugin or the legacy ORT wheel")
    return plugin


def register_qnn_plugin(ort, plugin):
    """Register once; keep the library registered for the process/session lifetime."""
    if not callable(getattr(ort, "register_execution_provider_library", None)):
        raise RuntimeError("This ORT lacks plugin EP registration; install an ORT version matching QNN 2.x")
    library = Path(plugin.get_library_path()).resolve()
    if not library.is_file():
        raise FileNotFoundError(f"QNN plugin library is missing: {library}")
    print(f"QNN plugin: {getattr(plugin, '__version__', 'unknown')} at {plugin.__file__}")
    print(f"QNN EP library: {library}")
    ort.register_execution_provider_library("QNNExecutionProvider", str(library))
    devices = [d for d in ort.get_ep_devices() if d.ep_name == "QNNExecutionProvider"]
    if not devices:
        raise RuntimeError("QNN plugin registered but no QNN device was discovered; check plugin, HTP libraries and BSP")
    print(f"Discovered QNN devices: {len(devices)}")
    return devices


def configure_loader(backend, plugin_library=None):
    """Re-exec before importing ORT so Linux's loader sees the new search path."""
    if not sys.platform.startswith("linux"):
        return
    env = os.environ.copy()
    libraries = [p for p in env.get("LD_LIBRARY_PATH", "").split(":") if p]
    preferred = [str(backend.parent)]
    if plugin_library is not None and str(plugin_library.parent) not in preferred:
        preferred.append(str(plugin_library.parent))
    updated = preferred + [p for p in libraries if p not in preferred]
    changed = updated != libraries
    if changed:
        env["LD_LIBRARY_PATH"] = ":".join(updated)
    if not env.get("ADSP_LIBRARY_PATH"):
        # Prefer skels shipped beside the selected plugin runtime over older apt libraries.
        skels = list(backend.parent.rglob("*Skel.so")) if plugin_library is not None else []
        if not skels:
            skels = [p for p in package_files("qairt-dsp-binaries")
                     if "Skel" in p.name and p.suffix == ".so" and p.is_file()]
        dsp_dirs = sorted({str(p.parent) for p in skels})
        if dsp_dirs:
            env["ADSP_LIBRARY_PATH"] = ";".join(dsp_dirs + ["/usr/lib/rfsa/adsp", "/dsp"])
            changed = True
    if changed:
        print("Configuring QAIRT library search paths and restarting Python...", flush=True)
        os.execve(sys.executable, [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]], env)


def dtype_for(type_name, np):
    types = {
        "tensor(float)": np.float32, "tensor(float16)": np.float16,
        "tensor(double)": np.float64, "tensor(int64)": np.int64,
        "tensor(int32)": np.int32, "tensor(int16)": np.int16,
        "tensor(int8)": np.int8, "tensor(uint64)": np.uint64,
        "tensor(uint32)": np.uint32, "tensor(uint16)": np.uint16,
        "tensor(uint8)": np.uint8, "tensor(bool)": np.bool_,
    }
    if type_name not in types:
        raise ValueError(f"Unsupported input type: {type_name}")
    return np.dtype(types[type_name])


def make_inputs(session, input_dir, random, rng, np):
    feeds = {}
    for meta in session.get_inputs():
        dtype = dtype_for(meta.type, np)
        if random:
            if any(not isinstance(d, int) or d < 0 for d in meta.shape):
                raise ValueError(
                    f"{meta.name} has dynamic shape {meta.shape}; supply a real .npy input"
                )
            if np.issubdtype(dtype, np.floating):
                # Additive attention masks: zero means no masking.
                value = (np.zeros(meta.shape, dtype=dtype) if "mask" in meta.name.lower()
                         else (rng.standard_normal(meta.shape) * 0.01).astype(dtype))
            else:
                # Zero is valid for typical token indices. Random integer values
                # could exceed an embedding table or denote invalid indices.
                value = np.zeros(meta.shape, dtype=dtype)
        else:
            path = input_dir / f"{meta.name}.npy"
            if not path.is_file():
                raise FileNotFoundError(f"Missing input tensor: {path}")
            value = np.load(path, allow_pickle=False)
            if value.dtype != dtype:
                raise ValueError(f"{meta.name}: expected {dtype}, got {value.dtype}")
            if value.ndim != len(meta.shape) or any(
                isinstance(d, int) and d != actual
                for d, actual in zip(meta.shape, value.shape)
            ):
                raise ValueError(f"{meta.name}: expected {meta.shape}, got {value.shape}")
        # Preserve scalar rank, unlike ascontiguousarray() on a scalar.
        feeds[meta.name] = value if value.flags.c_contiguous else np.ascontiguousarray(value)
    return feeds


def create_session(ort, path, args):
    options = ort.SessionOptions()
    providers = None
    run_options = None
    ep_options = {"backend_path": str(args.backend.resolve())} if args.provider == "qnn" else {}
    if getattr(args, "qnn_profile", "off") != "off":
        profile_dir = args.output_dir / "profiles"
        profile_dir.mkdir(parents=True, exist_ok=True)
        ep_options.update(profiling_level=args.qnn_profile,
                          profiling_file_path=str((profile_dir / f"{path.stem}_qnn.csv").resolve()))
    if getattr(args, "ort_profile", False):
        profile_dir = args.output_dir / "profiles"
        profile_dir.mkdir(parents=True, exist_ok=True)
        options.enable_profiling = True
        options.profile_file_prefix = str((profile_dir / f"{path.stem}_ort").resolve())
    if args.provider == "qnn":
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        perf_mode = getattr(args, "performance_mode", "burst")
        ep_options["htp_performance_mode"] = perf_mode
        if getattr(args, "qnn_devices", None) is not None:
            options.add_provider_for_devices(args.qnn_devices, ep_options)
            run_options = ort.RunOptions()
            run_options.add_run_config_entry("qnn.perf_mode", perf_mode)
            if getattr(args, "rpc_control_latency", None) is not None:
                run_options.add_run_config_entry("qnn.rpc_control_latency", str(args.rpc_control_latency))
        else:
            if getattr(args, "rpc_control_latency", None) is not None:
                ep_options["rpc_control_latency"] = str(args.rpc_control_latency)
            providers = [("QNNExecutionProvider", ep_options)]
    else:
        providers = ["CPUExecutionProvider"]
    started = time.perf_counter()
    session = ort.InferenceSession(str(path.resolve()), sess_options=options, providers=providers)
    session.disable_fallback()
    load_ms = (time.perf_counter() - started) * 1000
    if args.provider == "qnn" and "QNNExecutionProvider" not in session.get_providers():
        raise RuntimeError("QNN EP did not activate")
    print(f"\nModel: {path}\nProviders: {session.get_providers()}\nLoad: {load_ms:.2f} ms")
    if args.provider == "qnn":
        print(f"QNN options: {ep_options}; RPC latency: {getattr(args, 'rpc_control_latency', None)}")
    for meta in session.get_inputs():
        print(f"  INPUT  {meta.name}: {meta.shape} {meta.type}")
    for meta in session.get_outputs():
        print(f"  OUTPUT {meta.name}: {meta.shape} {meta.type}")
    return session, run_options, load_ms


def run_model(ort, np, path, args):
    session, run_options, load_ms = create_session(ort, path, args)
    if args.inspect:
        return
    feeds = make_inputs(session, args.input_dir, args.random, np.random.default_rng(args.seed), np)
    for _ in range(args.warmup):
        session.run(None, feeds, run_options)
    times = []
    for _ in range(args.repeat):
        started = time.perf_counter()
        outputs = session.run(None, feeds, run_options)
        times.append((time.perf_counter() - started) * 1000)
    if getattr(args, "ort_profile", False):
        print(f"ORT trace: {session.end_profiling()}")
    out_dir = args.output_dir / path.stem
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {out_dir}; choose --output-dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for i, (meta, value) in enumerate(zip(session.get_outputs(), outputs)):
        finite = bool(np.isfinite(value).all())
        # Numeric filenames avoid output tensor names that contain slashes.
        filename = f"output_{i}.npy"
        np.save(out_dir / filename, value)
        summary.append({"name": meta.name, "file": filename,
                        "shape": list(value.shape), "dtype": str(value.dtype), "finite": finite})
        print(f"  {meta.name}: shape={value.shape}, dtype={value.dtype}, finite={finite}")
    report = {
        "model": str(path.resolve()), "onnxruntime": ort.__version__,
        "providers": session.get_providers(), "random_inputs": args.random,
        "load_ms": load_ms, "mean_inference_ms": float(np.mean(times)),
        "iterations_ms": times, "outputs": summary,
        "warmup": args.warmup, "repeat": args.repeat,
        "performance_mode": getattr(args, "performance_mode", "burst"),
        "rpc_control_latency": getattr(args, "rpc_control_latency", None),
        "profiling_enabled": bool(getattr(args, "ort_profile", False)
                                  or getattr(args, "qnn_profile", "off") != "off"),
        "io_bytes": {"inputs": sum(v.nbytes for v in feeds.values()),
                     "outputs": sum(v.nbytes for v in outputs)},
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Mean inference: {report['mean_inference_ms']:.2f} ms; saved: {out_dir}")
    print(f"Input/output bytes per call: {report['io_bytes']}")
    if not all(item["finite"] for item in summary):
        raise RuntimeError("Non-finite outputs; see saved output tensors. Random input may be invalid for this graph.")


def pipeline_paths(paths):
    result = {}
    for name in ("vit", "llm", "vlm_proj", "dit"):
        matches = [p for p in paths if re.search(
            rf"(^|[_.\-]){name}($|[_.\-])", p.stem.lower(),
        )]
        if len(matches) != 1:
            raise ValueError(f"Pipeline requires exactly one {name} ONNX, found: {matches}. "
                             "Use four explicit --model arguments to avoid duplicate models.")
        result[name] = matches[0]
    return result


def check_feeds(session, feeds, np):
    metas = {m.name: m for m in session.get_inputs()}
    if set(metas) != set(feeds):
        raise ValueError(f"Input names mismatch: expected {list(metas)}, got {list(feeds)}")
    for name, meta in metas.items():
        value = feeds[name]
        if value.dtype != dtype_for(meta.type, np):
            raise ValueError(f"{name}: expected {meta.type}, got {value.dtype}")
        if value.ndim != len(meta.shape) or any(
            isinstance(d, int) and d != actual
            for d, actual in zip(meta.shape, value.shape)
        ):
            raise ValueError(f"{name}: expected {meta.shape}, got {value.shape}")


def pipeline_step(sessions, run_options, base, np):
    """Actually pass outputs between all four graphs; DiT includes denoising."""
    elapsed = {}

    def call(name, feeds):
        sess = sessions[name]
        check_feeds(sess, feeds, np)
        output_names = [m.name for m in sess.get_outputs()]
        started = time.perf_counter()
        values = sess.run(output_names, feeds, run_options[name])
        elapsed[name] = (time.perf_counter() - started) * 1000
        return dict(zip(output_names, values))

    vision = call("vit", {"pixel_values": base["pixel_values"]})["vit_embeds"]
    embeds = base["input_embeds"].copy()
    if embeds.ndim != 3 or vision.ndim != 3 or embeds.shape[-1] != vision.shape[-1]:
        raise ValueError(f"ViT/LLM shape mismatch: {vision.shape}/{embeds.shape}")
    selected = base["image_token_mask"]
    if selected.dtype != np.bool_ or selected.shape != embeds.shape[:2]:
        raise ValueError("image_token_mask must be boolean and match input_embeds' first two dimensions")
    flat = vision.reshape(-1, embeds.shape[-1])
    if int(selected.sum()) != flat.shape[0]:
        raise ValueError("Image token count does not match ViT output")
    embeds[selected] = flat
    vlm = call("llm", {"input_embeds": embeds,
                       "llm_attention_mask": base["llm_attention_mask"]})["vlm_embeds"]
    kv = call("vlm_proj", {"vlm_embeds": vlm,
                           "vlm_attention_mask": base["vlm_attention_mask"]})
    actions = call("dit", {
        "state": base["state"], "actions": base["actions"],
        "cross_attention_mask": base["cross_attention_mask"], **kv,
    })["actions_out"]
    return actions, elapsed


def make_simulated_image(np):
    """A deterministic 640x480 RGB test scene: table, colored blocks, mock gripper."""
    image = np.full((480, 640, 3), (210, 224, 238), dtype=np.uint8)
    image[240:] = (165, 125, 85)
    image[275:365, 100:210] = (220, 45, 45)
    image[300:390, 360:480] = (40, 90, 210)
    image[95:240, 290:330] = (70, 75, 85)
    image[220:280, 265:285] = (80, 85, 95)
    image[220:280, 335:355] = (80, 85, 95)
    return image


def simulation_setup(base, args, np):
    from PIL import Image

    config = {
        "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5],
        "state_mean": [0.0] * 14, "state_std": [1.0] * 14,
        "action_min": [-2.0] * 14, "action_max": [2.0] * 14,
        "action_step_ms": 50.0,
    }
    if args.simulation_config:
        config.update(json.loads(args.simulation_config.read_text(encoding="utf-8")))
    mean = np.asarray(config["image_mean"], np.float32)
    std = np.asarray(config["image_std"], np.float32)
    state_mean = np.asarray(config["state_mean"], np.float32)
    state_std = np.asarray(config["state_std"], np.float32)
    low = np.asarray(config["action_min"], np.float32)
    high = np.asarray(config["action_max"], np.float32)
    if mean.shape != (3,) or std.shape != (3,) or np.any(std <= 0):
        raise ValueError("image_mean/std must be three values, with positive std")
    if state_mean.ndim != 1 or state_mean.shape != state_std.shape or np.any(state_std <= 0):
        raise ValueError("state_mean/std must have matching 1D shapes, with positive std")
    if low.ndim != 1 or low.size < 1 or low.shape != high.shape or np.any(high <= low):
        raise ValueError("action_min/max must have matching nonempty 1D shapes and max > min")
    if not all(np.isfinite(a).all() for a in (mean, std, state_mean, state_std, low, high)):
        raise ValueError("Simulation configuration must contain finite values")
    if state_mean.size > base["state"].shape[-1] or low.size > base["actions"].shape[-1]:
        raise ValueError("Configured active dimensions exceed model's state/action dimensions")
    if not np.isfinite(config["action_step_ms"]) or config["action_step_ms"] <= 0:
        raise ValueError("action_step_ms must be positive")
    if base["pixel_values"].shape[:2] != (1, 3):
        raise ValueError("Image simulation currently requires ViT [1, 3, H, W] NCHW")
    image = Image.fromarray(make_simulated_image(np))
    out_dir = args.output_dir / "pipeline"
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    image.save(out_dir / "simulated_image.png")
    raw_state = np.linspace(-0.2, 0.2, state_mean.size, dtype=np.float32)
    return {"image": image, "config": config, "mean": mean, "std": std,
            "state_mean": state_mean, "state_std": state_std, "raw_state": raw_state,
            "low": low, "high": high, "rng": np.random.default_rng(args.seed)}


def simulation_preprocess(base, context, np):
    from PIL import Image

    feeds = dict(base)
    _, _, height, width = base["pixel_values"].shape
    resized = context["image"].resize((width, height), Image.Resampling.BILINEAR)
    rgb = np.asarray(resized, dtype=np.float32) / np.float32(255.0)
    pixels = (rgb - context["mean"]) / context["std"]
    feeds["pixel_values"] = np.ascontiguousarray(pixels.transpose(2, 0, 1)[None])
    state = np.zeros_like(base["state"])
    count = context["state_mean"].size
    state[..., :count] = (context["raw_state"] - context["state_mean"]) / context["state_std"]
    feeds["state"] = state
    # All synthetic token positions are valid; construct an additive causal mask.
    mask = np.zeros_like(base["llm_attention_mask"])
    rows, columns = mask.shape[-2:]
    future = np.arange(columns)[None, :] > np.arange(rows)[:, None]
    mask[..., future] = np.finfo(mask.dtype).min
    feeds["llm_attention_mask"] = mask
    feeds["vlm_attention_mask"] = np.zeros_like(base["vlm_attention_mask"])
    feeds["cross_attention_mask"] = np.zeros_like(base["cross_attention_mask"])
    feeds["actions"] = context["rng"].standard_normal(base["actions"].shape).astype(np.float32)
    return feeds


def simulation_postprocess(raw, context, np):
    """Mock min/max unnormalization and padded-dimension removal, not robot commands."""
    if not np.isfinite(raw).all():
        raise RuntimeError("Non-finite model actions")
    active = raw[..., :context["low"].size]
    # Clipping is a simulation assumption, not a claimed GR00T checkpoint transform.
    values = np.clip(active, -1.0, 1.0)
    actions = (values + np.float32(1.0)) * np.float32(0.5)
    actions = actions * (context["high"] - context["low"]) + context["low"]
    return np.ascontiguousarray(actions)


def run_pipeline(ort, np, args):
    paths = pipeline_paths(args.model)
    sessions, options, loads = {}, {}, {}
    # Only these four calls create sessions; warmup and all measured iterations reuse them.
    for name, path in paths.items():
        sessions[name], options[name], loads[name] = create_session(ort, path, args)
    if args.pipeline_inputs:
        with np.load(args.pipeline_inputs, allow_pickle=False) as archive:
            base = {k: archive[k] for k in archive.files}
        print("Prepared input tensors: image/token/state preprocessing is outside this benchmark.")
    else:
        print("SYNTHETIC FOUR-GRAPH PIPELINE: fixed random inputs, no real task semantics.")
        rng = np.random.default_rng(args.seed)
        initial = {k: make_inputs(s, None, True, rng, np) for k, s in sessions.items()}
        base = {"pixel_values": initial["vit"]["pixel_values"],
                "input_embeds": initial["llm"]["input_embeds"],
                "llm_attention_mask": initial["llm"]["llm_attention_mask"],
                "vlm_attention_mask": initial["vlm_proj"]["vlm_attention_mask"],
                "state": initial["dit"]["state"],
                "actions": rng.standard_normal(initial["dit"]["actions"].shape).astype(np.float32),
                "cross_attention_mask": initial["dit"]["cross_attention_mask"]}
        shape = next(m.shape for m in sessions["vit"].get_outputs() if m.name == "vit_embeds")
        if len(shape) != 3 or not all(isinstance(d, int) for d in shape):
            raise ValueError("Synthetic mode requires a static [batch, tokens, hidden] ViT output")
        batch, tokens, hidden = shape
        if base["input_embeds"].shape[0] != batch or base["input_embeds"].shape[2] != hidden:
            raise ValueError("Synthetic mode requires matching ViT and LLM batch/hidden dimensions")
        if tokens > base["input_embeds"].shape[1]:
            raise ValueError("ViT has more tokens than the LLM input")
        selected = np.zeros(base["input_embeds"].shape[:2], dtype=np.bool_)
        selected[:, :tokens] = True  # Synthetic layout only; no tokenizer assumptions for real inputs.
        base["image_token_mask"] = selected
        del initial
    simulation = simulation_setup(base, args, np) if getattr(args, "simulate_image", False) else None

    def execute():
        started = time.perf_counter()
        feeds = simulation_preprocess(base, simulation, np) if simulation is not None else base
        pre_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        raw, measured = pipeline_step(sessions, options, feeds, np)
        model_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        decoded = simulation_postprocess(raw, simulation, np) if simulation is not None else None
        post_ms = (time.perf_counter() - started) * 1000
        return raw, decoded, measured, pre_ms, model_ms, post_ms

    for i in range(args.warmup):
        actions, _, _, _, _, _ = execute()
        if not np.isfinite(actions).all():
            raise RuntimeError(f"Non-finite warmup output at iteration {i + 1}")
    times = []
    stages = {"preprocess": [], "model_pipeline": [], "postprocess": []}
    per_component = {k: [] for k in sessions}
    print(f"Warmup finished. Measuring {args.repeat} complete pipeline iterations...", flush=True)
    for i in range(args.repeat):
        started = time.perf_counter()
        actions, decoded, measured, pre_ms, model_ms, post_ms = execute()
        times.append((time.perf_counter() - started) * 1000)
        for name, value in zip(stages, (pre_ms, model_ms, post_ms)):
            stages[name].append(value)
        if not np.isfinite(actions).all():
            raise RuntimeError(f"Non-finite output at measured iteration {i + 1}")
        for name, value in measured.items():
            per_component[name].append(value)
        if (i + 1) % 10 == 0 or i + 1 == args.repeat:
            print(f"  {i + 1}/{args.repeat}: mean {np.mean(times):.2f} ms", flush=True)
    if getattr(args, "ort_profile", False):
        for name, sess in sessions.items():
            print(f"ORT trace ({name}): {sess.end_profiling()}")
    report = {
        "mode": "four_graph_pipeline", "synthetic_inputs": not bool(args.pipeline_inputs),
        "onnxruntime": ort.__version__, "repeat": args.repeat, "warmup": args.warmup,
        "models": {k: str(p.resolve()) for k, p in paths.items()},
        "providers": {k: s.get_providers() for k, s in sessions.items()},
        "load_ms": loads, "total_load_ms": float(sum(loads.values())),
        "mean_pipeline_ms": float(np.mean(times)), "median_pipeline_ms": float(np.median(times)),
        "min_pipeline_ms": float(np.min(times)), "max_pipeline_ms": float(np.max(times)),
        "p95_pipeline_ms": float(np.percentile(times, 95)),
        "total_measured_ms": float(sum(times)), "iterations_ms": times,
        "component_mean_ms": {k: float(np.mean(v)) for k, v in per_component.items()},
        "component_iterations_ms": per_component,
        "simulated_image": simulation is not None,
        "stage_mean_ms": {k: float(np.mean(v)) for k, v in stages.items()},
        "stage_iterations_ms": stages,
        "performance_mode": getattr(args, "performance_mode", "burst"),
        "rpc_control_latency": getattr(args, "rpc_control_latency", None),
        "profiling_enabled": bool(getattr(args, "ort_profile", False)
                                  or getattr(args, "qnn_profile", "off") != "off"),
        "scope": "Four Session.run calls, input validation and intermediate tensor assembly; "
                 "excludes session loading, warmup, raw-input preprocessing, action unnormalization and output saving.",
    }
    out_dir = args.output_dir / "pipeline"
    if simulation is None and out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "actions_normalized.npy", actions)
    if simulation is not None:
        report["scope"] = ("Synthetic RGB resize/normalization, masks/state/noise preparation, "
                          "four connected models and mock action unnormalization; excludes image "
                          "generation, loading, warmup, tokenizer/real text embedding and file saving.")
        report["simulation_config"] = simulation["config"]
        report["action_warning"] = "Simulation values only; not validated robot commands."
        np.save(out_dir / "actions_simulated.npy", decoded)
        timestamps = np.arange(decoded.shape[1]) * simulation["config"]["action_step_ms"]
        np.savetxt(out_dir / "actions_simulated.csv", np.column_stack((timestamps, decoded[0])),
                   delimiter=",", header="time_ms," + ",".join(f"sim_joint_{i}" for i in range(decoded.shape[-1])),
                   comments="")
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nLoad once: {report['total_load_ms']:.2f} ms")
    print(f"PIPELINE MEAN: {report['mean_pipeline_ms']:.2f} ms")
    print(f"Median: {report['median_pipeline_ms']:.2f} ms; P95: {report['p95_pipeline_ms']:.2f} ms")
    print(f"Component means: {report['component_mean_ms']}")
    print(f"Stage means: {report['stage_mean_ms']}")
    print(f"Saved report and normalized actions: {out_dir}")


def diagnostic_snapshot():
    result = {"python": sys.executable, "platform": sys.platform,
              "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH"),
              "ADSP_LIBRARY_PATH": os.environ.get("ADSP_LIBRARY_PATH"), "packages": {}, "sysfs": {}}
    for name in ("onnxruntime", "onnxruntime-qnn", "numpy"):
        try:
            result["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result["packages"][name] = None
    # Read-only snapshots. Sysfs layouts and access vary with the BSP.
    patterns = (
        "/sys/class/thermal/thermal_zone*/temp", "/sys/class/thermal/thermal_zone*/type",
        "/sys/devices/system/cpu/cpufreq/policy*/scaling_cur_freq",
        "/sys/devices/system/cpu/cpufreq/policy*/scaling_governor",
        "/sys/class/devfreq/*/cur_freq", "/sys/class/devfreq/*/governor",
    )
    import glob
    for pattern in patterns:
        for filename in glob.glob(pattern):
            try:
                result["sysfs"][filename] = Path(filename).read_text().strip()
            except OSError:
                pass
    try:
        result["load_average"] = list(os.getloadavg())
    except (AttributeError, OSError):
        pass
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    models = parser.add_mutually_exclusive_group()
    models.add_argument("--model", type=Path, action="append",
                        help="ONNX path; repeat for independent component checks")
    models.add_argument("--models-dir", type=Path, help="Find all ONNX files recursively; default: ./models")
    parser.add_argument("--backend", type=Path, help="QNN backend path; auto-detected when omitted")
    parser.add_argument("--qnn-plugin-dir", type=Path,
                        help="Optional onnxruntime_qnn package path; otherwise import from the active environment")
    parser.add_argument("--provider", choices=("qnn", "cpu"), default="qnn")
    parser.add_argument("--performance-mode", default="burst", choices=(
        "burst", "balanced", "default", "high_performance", "sustained_high_performance"))
    parser.add_argument("--rpc-control-latency", type=int, help="Optional QNN RPC latency in microseconds")
    parser.add_argument("--diagnostics", action="store_true", help="Save read-only board state before/after inference")
    parser.add_argument("--qnn-profile", choices=("off", "basic", "detailed"), default="off")
    parser.add_argument("--ort-profile", action="store_true", help="Save an ORT execution trace (adds profiling overhead)")
    parser.add_argument("--pipeline", action="store_true", help="Chain ViT -> LLM -> projection -> DiT, loading each once")
    parser.add_argument("--pipeline-inputs", type=Path, help="Optional prepared GR00T NPZ; otherwise synthetic pipeline inputs")
    parser.add_argument("--simulate-image", action="store_true", help="Include synthetic RGB preprocessing and mock action postprocessing in pipeline timing")
    parser.add_argument("--simulation-config", type=Path, help="Optional JSON image/state/action parameters for the simulation")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--inspect", action="store_true", help="Load model and show I/O only")
    mode.add_argument("--random", action="store_true", help="Synthetic smoke test; outputs have no task meaning")
    mode.add_argument("--input-dir", type=Path, help="Directory of <input_tensor_name>.npy files")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("ort_output") / time.strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, help="Measured iterations; default 100 for pipeline, 3 for independent models")
    args = parser.parse_args()
    if args.repeat is None:
        args.repeat = 100 if args.pipeline else 3
    if args.repeat < 1 or args.warmup < 0:
        parser.error("--repeat must be >=1; --warmup must be >=0")
    if args.rpc_control_latency is not None and args.rpc_control_latency < 0:
        parser.error("--rpc-control-latency must be >=0")
    if args.qnn_profile != "off" and args.provider != "qnn":
        parser.error("--qnn-profile requires --provider qnn")
    if args.pipeline and (args.inspect or args.input_dir):
        parser.error("--pipeline cannot be used with --inspect or --input-dir; use --pipeline-inputs")
    if args.pipeline_inputs and (not args.pipeline or args.random):
        parser.error("--pipeline-inputs requires --pipeline and cannot be used with --random")
    if args.simulate_image and (not args.pipeline or args.pipeline_inputs):
        parser.error("--simulate-image requires --pipeline and cannot use --pipeline-inputs")
    if args.simulation_config and not args.simulate_image:
        parser.error("--simulation-config requires --simulate-image")
    if not args.inspect and args.input_dir is None:
        args.random = True
    if not args.model:
        folder = args.models_dir or Path("models")
        args.model = sorted(folder.rglob("*.onnx"))
        if not args.model:
            parser.error(f"No ONNX models found in {folder}; use --model or --models-dir")
    if any(not path.is_file() for path in args.model):
        parser.error("An ONNX file does not exist")
    if len({path.stem for path in args.model}) != len(args.model):
        parser.error("Multiple models with identical filenames: run each separately")
    plugin = None
    args.qnn_devices = None
    if args.provider == "qnn":
        try:
            plugin = load_qnn_plugin(args.qnn_plugin_dir)
            if plugin is not None and args.backend is None:
                args.backend = Path(plugin.get_qnn_htp_path())
            args.backend = find_backend(args.backend)
        except (ImportError, FileNotFoundError, ValueError, RuntimeError) as error:
            parser.error(str(error))
        print(f"QNN backend: {args.backend}", flush=True)
        plugin_library = Path(plugin.get_library_path()).resolve() if plugin is not None else None
        configure_loader(args.backend, plugin_library)
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError as error:
        parser.error(f"Missing dependency: {error}. Install NumPy and the board's QNN-enabled ORT wheel.")

    provider = "QNNExecutionProvider" if args.provider == "qnn" else "CPUExecutionProvider"
    print(f"ORT {ort.__version__}; available: {ort.get_available_providers()}")
    if plugin is not None:
        args.qnn_devices = register_qnn_plugin(ort, plugin)
    elif provider not in ort.get_available_providers():
        raise RuntimeError(f"This ONNX Runtime does not include {provider}")
    diagnostics = {"before": diagnostic_snapshot()} if args.diagnostics else None
    if args.ort_profile or args.qnn_profile != "off":
        print("PROFILING ENABLED: timings include diagnostic overhead; compare against an unprofiled run.")
    try:
        if args.pipeline:
            run_pipeline(ort, np, args)
        else:
            if args.random:
                print("RANDOM INPUT SMOKE TEST: each graph runs independently; outputs are not robot commands.")
            for path in args.model:
                run_model(ort, np, path, args)
    finally:
        if diagnostics is not None:
            diagnostics["after"] = diagnostic_snapshot()
            diagnostics["backend"] = str(args.backend) if args.backend else None
            diagnostics["plugin"] = getattr(plugin, "__file__", None)
            args.output_dir.mkdir(parents=True, exist_ok=True)
            dest = args.output_dir / "diagnostics.json"
            dest.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
            print(f"Diagnostics: {dest}")


if __name__ == "__main__":
    main()


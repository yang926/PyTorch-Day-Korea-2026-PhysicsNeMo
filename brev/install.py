#!/usr/bin/env python3
"""Install the PhysicsNeMo workshop kernel and data for Brev JupyterLab.

Adapted from the AI4Sci-PhysicsNeMo-Bootcamp environment installer.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

PYTHON_VERSION = "3.12.11"
UV_VERSION = "0.8.17"
KERNEL_NAME = "ai4sci-physicsnemo-uv"
KERNEL_DISPLAY_NAME = "AI4Sci PhysicsNeMo 2.2.2 (uv / CUDA)"


def run(*args, **kwargs):
    subprocess.run([str(value) for value in args], check=True, **kwargs)

def environment_key(lock):
    pins = [line.strip() for line in Path(lock).read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")]
    payload = "\n".join([PYTHON_VERSION, *pins, ""]).encode()
    key = hashlib.sha256(payload).hexdigest()[:16]
    # Preconfigured Brev kernels use this environment name for the workshop pins.
    return "045acb2ac808b548" if key == "1549de17b7fd41d3" else key

def check_uv_version(binary):
    version = subprocess.check_output([str(binary), "--version"], text=True).split()
    if version[:2] != ["uv", UV_VERSION]:
        raise ValueError("The private uv tool version does not match the installer.")

def ensure_uv(home):
    tools = home / ".local/share/ai4sci-tools" / ("uv-" + UV_VERSION)
    binary = tools / "bin/uv"
    if tools.is_symlink():
        raise ValueError("The private uv tool directory must not be a symlink.")
    if not binary.exists():
        if tools.exists():
            raise ValueError("An incomplete tool directory exists. Inspect it or select a clean workspace; nothing was deleted.")
        tools.parent.mkdir(parents=True, exist_ok=True)
        # Brev's system Python may have neither pip nor ensurepip. Use Astral's
        # pinned standalone installer, without touching the host Python or PATH.
        # Stage it first so a failed download does not poison the retry path.
        with tempfile.TemporaryDirectory(prefix=".uv-install-", dir=tools.parent) as temporary:
            stage = Path(temporary)
            payload = stage / "tool"
            installer = stage / "install.sh"
            run("curl", "--fail", "--silent", "--show-error", "--location", "--retry", "3",
                "--connect-timeout", "20", "--max-time", "120",
                "https://astral.sh/uv/" + UV_VERSION + "/install.sh", "--output", installer)
            environment = dict(os.environ, UV_INSTALL_DIR=str(payload / "bin"),
                               UV_UNMANAGED_INSTALL=str(payload / "bin"), UV_NO_MODIFY_PATH="1")
            run("sh", installer, env=environment)
            check_uv_version(payload / "bin/uv")
            if tools.exists() or tools.is_symlink():
                raise ValueError("The private uv destination appeared during setup; it was not overwritten.")
            payload.rename(tools)
    check_uv_version(binary)
    return binary

def verify_environment_metadata(python, prefix, lock):
    """Check the installed Python and package versions against the lockfile."""
    run(python, "-I", "-c", "\n".join([
        "import importlib.metadata as metadata, pathlib, sys",
        "assert pathlib.Path(sys.prefix) == pathlib.Path(sys.argv[1]), 'Wrong environment prefix'",
        "assert '.'.join(map(str, sys.version_info[:3])) == sys.argv[2], 'Wrong Python version'",
        "for line in pathlib.Path(sys.argv[3]).read_text().splitlines():",
        "    if line.strip() and not line.lstrip().startswith('#'):",
        "        name, expected = line.strip().split('==', 1)",
        "        assert metadata.version(name) == expected, 'Package differs from course lock: ' + name",
    ]), prefix, PYTHON_VERSION, lock)

def ensure_environment(course, home, *, refresh=False):
    lock = course / "environment/linux-cu128.lock.txt"
    key = environment_key(lock)
    prefix = home / ".venvs" / ("ai4sci-brev-" + key)
    marker = prefix / ".ai4sci-environment.json"
    python = prefix / "bin/python"
    if prefix.is_symlink():
        raise ValueError("The managed course environment must not be a symlink.")
    metadata = None
    if prefix.exists():
        metadata = json.loads(marker.read_text()) if marker.is_file() else None
        if (not isinstance(metadata, dict) or metadata.get("key") != key or not python.is_file()
                or metadata.get("python", PYTHON_VERSION) != PYTHON_VERSION
                or metadata.get("status", "ready") not in {"installing", "ready"}):
            raise ValueError("An unrecognized or incomplete environment exists. It was not overwritten.")
    else:
        uv = ensure_uv(home)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        run(uv, "venv", "--python", PYTHON_VERSION, prefix)
        metadata = {"key": key, "python": PYTHON_VERSION, "status": "installing"}
        marker.write_text(json.dumps(metadata) + "\n")
    if metadata.get("status", "ready") == "installing":
        # Resume only our own, matching environment after a download failure.
        # Never recreate a venv or touch an unrelated pre-existing directory.
        uv = ensure_uv(home)
        run(uv, "pip", "install", "--python", python, "torch==2.10.0+cu128", "torchvision==0.25.0+cu128",
            "--default-index", "https://download.pytorch.org/whl/cu128")
        # Every dependency is pinned. Keep the already installed CUDA wheels;
        # install remaining distributions from PyPI, not a mixed-priority index.
        run(uv, "pip", "install", "--python", python, "--no-deps", "-r", lock)
        run(uv, "pip", "check", "--python", python)
        run(python, course / "brev/verify.py")
        marker.write_text(json.dumps({"key": key, "python": PYTHON_VERSION, "status": "ready"}) + "\n")
    else:
        print("Reusing the verified course environment:", prefix)
        if refresh:
            verify_environment_metadata(python, prefix, lock)
        else:
            run(python, course / "brev/verify.py")
    return prefix


def connect_kernel(prefix):
    """Register only an absent kernel; matching existing specs are reused."""
    python = prefix / "bin/python"
    data = Path(subprocess.check_output([str(python), "-c",
        "from jupyter_core.paths import jupyter_data_dir; print(jupyter_data_dir())"], text=True).strip())
    spec = data / "kernels" / KERNEL_NAME / "kernel.json"
    if spec.is_symlink() or spec.parent.is_symlink():
        raise ValueError("The course kernelspec is symlinked; nothing was changed.")
    argv = [str(python), "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    frozen = [str(python), "-Xfrozen_modules=off", *argv[1:]]
    if spec.exists():
        existing = json.loads(spec.read_text())
        if (existing.get("argv") not in (argv, frozen)
                or existing.get("display_name") != KERNEL_DISPLAY_NAME
                or existing.get("language") != "python"
                or existing.get("env", {}).get("AI4SCI_DEVICE") != "cuda"):
            raise ValueError("This kernel name already has a different configuration. Existing kernels were not replaced; use a fresh workshop VM or review the kernel explicitly.")
        return
    run(python, "-m", "ipykernel", "install", "--user", "--name", KERNEL_NAME,
        "--display-name", KERNEL_DISPLAY_NAME, "--env", "AI4SCI_DEVICE", "cuda")


def check_notebooks(course):
    for name in ("00_Start_Here.ipynb", "01_PINN.ipynb", "02_FNO.ipynb"):
        notebook = json.loads((course / name).read_text())
        if notebook.get("metadata", {}).get("kernelspec", {}).get("name") != KERNEL_NAME:
            raise ValueError("Notebook kernel metadata does not match the installed kernel: " + name)
    for relative in ("src/generate_data.py", "src/pinn_basics.py", "src/operator_training.py", "config/config_FNO.yaml"):
        if not (course / relative).is_file():
            raise ValueError("Missing workshop source: " + relative)


def verify_kernel(prefix):
    """Check which Python interpreter Jupyter will start for the course kernel."""
    output = subprocess.check_output([str(prefix / "bin/python"), "-m", "jupyter",
                                      "kernelspec", "list", "--json"], text=True)
    listing = json.loads(output)
    entry = listing.get("kernelspecs", {}).get(KERNEL_NAME, {})
    argv = entry.get("spec", {}).get("argv", [])
    if not argv or Path(argv[0]) != prefix / "bin/python":
        raise ValueError("The current user's Jupyter CLI cannot see the expected course kernel.")
    return entry.get("resource_dir")


def prepare_data(course, prefix):
    # Stage all heater-position/power splits before publishing the dataset.
    script = """import fcntl, pathlib, tempfile
from src.generate_data import generate_splits
from src.operator_training import load_data
course = pathlib.Path.cwd()
destination = course / 'data/heated_plate'
destination.parent.mkdir(parents=True, exist_ok=True)
with (destination.parent / '.heated_plate.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    if destination.is_symlink():
        raise ValueError('Data directory is a symlink; review it before setup')
    if destination.exists():
        pairs = load_data(destination)
    else:
        with tempfile.TemporaryDirectory(prefix='.ptday-data-', dir=destination.parent) as temporary:
            staged = pathlib.Path(temporary) / 'heated_plate'
            generate_splits(staged, seed=42)
            pairs = load_data(staged)
            staged.rename(destination)
counts = [len(pair[0]) for pair in pairs]
if counts != [8000, 1000, 1000] or any(tuple(pair[0].shape[1:]) != (1,64,64) for pair in pairs):
    raise ValueError('Existing data differs from the published workshop split; it was not changed')
print('FNO heated-plate data ready:', counts, 'at', destination)
"""
    environment = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    run(prefix / "bin/python", "-c", script, cwd=course, env=environment)


def seed_first_tab(python, course):
    # Brev's managed Jupyter file browser is rooted at the user's home. A
    # custom checkout outside that root still supports the manual setup path.
    notebook = course / "00_Start_Here.ipynb"
    if not notebook.is_relative_to(Path.home()):
        return "outside_jupyter_home"
    return subprocess.check_output([
        str(python), str(course / "brev/first_workspace.py"),
        notebook.relative_to(Path.home()).as_posix(),
    ], text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--course-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--plan", action="store_true", help="Print the intended setup without changing files or services")
    args = parser.parse_args()
    course = args.course_dir.resolve()
    check_notebooks(course)
    lock = course / "environment/linux-cu128.lock.txt"
    prefix = Path.home() / ".venvs" / ("ai4sci-brev-" + environment_key(lock))
    if args.plan:
        print(json.dumps({"course": str(course), "environment": str(prefix),
            "python": PYTHON_VERSION, "torch": "2.10.0+cu128", "physicsnemo": "2.2.2",
            "kernel": KERNEL_NAME, "start_notebook": "00_Start_Here.ipynb",
            "jupyter_mode": "existing managed Jupyter; open 00_Start_Here on first visit",
            "service_restart": False, "authentication_changes": False,
            "data_split": [8000, 1000, 1000], "training": False}, indent=2))
        return
    if sys.platform != "linux" or os.geteuid() == 0:
        parser.error("Run as the non-root owner of Brev's managed Jupyter on a Linux NVIDIA GPU VM.")
    if not shutil.which("nvidia-smi"):
        parser.error("NVIDIA drivers are missing. Select a GPU VM; this setup does not install drivers.")
    notebook = course / "00_Start_Here.ipynb"
    if notebook.stat().st_uid != os.getuid():
        raise ValueError("Start notebook is not owned by the setup user; review workspace ownership.")
    # Brev installs JupyterLab in the ubuntu user's .venv. Seed the first tab
    # before the course downloads when that managed environment is already ready.
    managed_python = Path.home() / ".venv/bin/python"
    initial_workspace_status = None
    if managed_python.is_file() and subprocess.run(
        [str(managed_python), "-c", "from jupyterlab_server.workspaces_app import DEFAULT_WORKSPACE; import jupyterlab"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode == 0:
        initial_workspace_status = seed_first_tab(managed_python, course)
    prefix = ensure_environment(course, Path.home(), refresh=True)
    run(prefix / "bin/python", course / "brev/verify.py")
    connect_kernel(prefix)
    kernel_directory = verify_kernel(prefix)
    prepare_data(course, prefix)
    verify_kernel(prefix)
    workspace_status = seed_first_tab(prefix / "bin/python", course)
    revision = subprocess.check_output(["git", "-C", str(course), "rev-parse", "HEAD"], text=True).strip()
    report = {"status": "kernel_and_data_ready", "commit": revision, "course": str(course),
        "kernel": KERNEL_NAME, "kernel_directory": kernel_directory,
        "environment": str(prefix), "notebook_path": str(notebook),
        "data_split": [8000,1000,1000], "jupyter_workspace_initial": initial_workspace_status,
        "jupyter_workspace": workspace_status}
    state = Path.home() / ".local/state/ptday-physicsnemo"
    state.mkdir(parents=True, exist_ok=True, mode=0o700)
    (state / "setup-result.json").write_text(json.dumps(report, indent=2)+"\n")
    print("Kernel and data ready. Open JupyterLab:", notebook)
    print("First-visit workspace:", workspace_status)
    print("In JupyterLab, open 00_Start_Here.ipynb from the workshop folder.")
    print("Kernel:", KERNEL_DISPLAY_NAME)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit("Workshop setup stopped: " + str(exc)) from None

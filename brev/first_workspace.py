"""Open the course notebook on the first visit to Brev's managed JupyterLab."""
import json
import os
from pathlib import Path, PurePosixPath
import sys
import tempfile


def unused_workspace(state, workspace_id):
    """Accept only Lab's untouched default or its first empty Launcher layout."""
    if not isinstance(state, dict) or not isinstance(state.get("metadata"), dict):
        return False
    if state["metadata"].get("id") != workspace_id:
        return False
    data = state.get("data")
    allowed = {"layout-restorer:data", "file-browser-filebrowser:columns",
               "file-browser-filebrowser:cwd", "docmanager:recents"}
    if not isinstance(data, dict) or set(data) - allowed:
        return False
    recents = data.get("docmanager:recents", {})
    if not isinstance(recents, dict) or any(recents.values()):
        return False
    layout = data.get("layout-restorer:data", {})
    if not isinstance(layout, dict):
        return False
    main = layout.get("main") or {}
    down = layout.get("down") or {}
    if not isinstance(main, dict) or not isinstance(down, dict) or down.get("widgets"):
        return False
    dock = main.get("dock") or {}
    if not isinstance(dock, dict) or dock.get("type") not in (None, "tab-area"):
        return False
    # Launcher widgets are not tracked as documents. Lab 4.6 saves widgets=[].
    return not main.get("current") and not dock.get("widgets")


def seed_workspace(notebook):
    from jupyterlab.commands import get_workspaces_dir
    from jupyter_core.paths import jupyter_runtime_dir
    from jupyterlab_server.workspaces_app import DEFAULT_WORKSPACE
    from jupyterlab_server.workspaces_handler import WorkspacesManager

    path = PurePosixPath(notebook)
    if path.is_absolute() or ".." in path.parts or path.suffix != ".ipynb":
        raise ValueError("Expected a notebook path relative to the Jupyter file browser.")
    folder = Path(get_workspaces_dir())
    folder.mkdir(parents=True, exist_ok=True)
    key = "notebook:" + path.as_posix()
    # These are the same tracker and main-tab records that Lab saves when a
    # user opens a notebook. No server settings, kernels or named workspaces change.
    workspace = {
        "metadata": {"id": DEFAULT_WORKSPACE},
        "data": {
            key: {"data": {"path": path.as_posix(), "factory": "Notebook"}},
            "layout-restorer:data": {"main": {
                "dock": {"type": "tab-area", "currentIndex": 0, "widgets": [key]},
                "current": key,
            }},
            "file-browser-filebrowser:cwd": {"path": path.parent.as_posix()},
        },
    }
    # Let the installed JupyterLab generate its own filename and validate the
    # workspace in isolation. The hard link atomically publishes only a new file.
    with tempfile.TemporaryDirectory(prefix=".ptday-", dir=folder) as stage:
        staged = WorkspacesManager(stage).save(DEFAULT_WORKSPACE, json.dumps(workspace))
        target = folder / staged.name
        try:
            os.link(staged, target)
        except FileExistsError:
            # Eagerly opening a new instance sometimes saves a blank Launcher
            # before setup finishes. Leave all established or active sessions alone.
            if target.is_symlink() or not target.is_file() or any(Path(jupyter_runtime_dir()).glob("kernel-*.json")):
                return "preserved_existing_workspace"
            before = target.read_bytes()
            try:
                prior = json.loads(before)
            except (ValueError, UnicodeDecodeError):
                return "preserved_existing_workspace"
            if not unused_workspace(prior, DEFAULT_WORKSPACE):
                return "preserved_existing_workspace"
            old_data = prior["data"]
            old_data[key] = workspace["data"][key]
            old_data.setdefault("layout-restorer:data", {})["main"] = workspace["data"]["layout-restorer:data"]["main"]
            old_data.setdefault("file-browser-filebrowser:cwd", {"path": path.parent.as_posix()})
            staged.write_text(json.dumps(prior), encoding="utf-8")
            # Claim only the snapshot we inspected. A concurrent save creating
            # a new target wins; never replace that path with the prepared seed.
            claimed = Path(stage) / "previous"
            try:
                target.rename(claimed)
            except FileNotFoundError:
                return "preserved_existing_workspace"
            try:
                if claimed.read_bytes() != before:
                    return "preserved_existing_workspace"
                try:
                    os.link(staged, target)
                except FileExistsError:
                    return "preserved_existing_workspace"
                return "seeded_empty_launcher"
            finally:
                try:
                    os.link(claimed, target)
                except FileExistsError:
                    pass
    return "seeded_first_notebook"


if __name__ == "__main__":
    print(seed_workspace(sys.argv[1]))

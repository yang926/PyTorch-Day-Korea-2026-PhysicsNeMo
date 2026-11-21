#!/usr/bin/env python3
"""Get the standalone workshop without changing an existing learner checkout."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

REPO = "https://github.com/yang926/PyTorch-Day-Korea-2026-PhysicsNeMo.git"
DIRECTORY = "PyTorch-Day-Korea-2026-PhysicsNeMo"

def git(*args):
    environment = dict(os.environ, GIT_TERMINAL_PROMPT="0")
    for name in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR",
                 "GIT_OBJECT_DIRECTORY", "GIT_ALTERNATE_OBJECT_DIRECTORIES",
                 "GIT_CONFIG_COUNT", "GIT_CONFIG_PARAMETERS"):
        environment.pop(name, None)
    return subprocess.check_output(["git", "-c", "core.hooksPath=/dev/null", *map(str,args)],
                                   text=True, env=environment).strip()

def target_commit(ref):
    if ref == "main":
        output = git("ls-remote", "--exit-code", REPO, "refs/heads/main")
        values = output.split()
        if len(values) != 2 or not re.fullmatch(r"[a-f0-9]{40}", values[0]):
            raise ValueError("Could not identify the published main commit.")
        return values[0]
    if not re.fullmatch(r"[a-f0-9]{40}",ref):
        raise ValueError("Ref must be main or a full lowercase commit SHA.")
    return ref

def prepare_checkout(destination, ref):
    target = target_commit(ref)
    if destination.is_symlink():
        raise ValueError("Workshop checkout is symlinked; nothing was changed.")
    if destination.exists():
        if Path(git("-C",destination,"rev-parse","--show-toplevel")).resolve() != destination.resolve():
            raise ValueError("The workshop destination is not its own Git checkout.")
        remote = git("-C",destination,"remote","get-url","origin")
        if remote.rstrip("/").removesuffix(".git") != REPO.removesuffix(".git"):
            raise ValueError("An unrelated checkout occupies the destination; it was preserved.")
        if git("-C",destination,"rev-parse","HEAD") != target:
            raise ValueError("Existing workshop revision differs from the requested release. Save learner work and use a fresh workspace; this script does not pull/reset/stash existing work.")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".ptday-clone-",dir=destination.parent) as temporary:
        stage=Path(temporary)/DIRECTORY
        git("clone","--single-branch","--branch","main",REPO,stage)
        if git("-C",stage,"rev-parse","HEAD") != target:
            git("-C",stage,"fetch","origin",target)
            git("-C",stage,"checkout","--detach",target)
        if git("-C",stage,"rev-parse","HEAD") != target:
            raise ValueError("Downloaded source does not match the requested commit.")
        if destination.exists() or destination.is_symlink():
            raise ValueError("Workshop destination appeared during setup; it was not replaced.")
        stage.rename(destination)
    return destination

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref",default=os.environ.get("PTDAY_COURSE_REF","main"))
    parser.add_argument("--plan",action="store_true")
    args=parser.parse_args()
    if args.ref != "main" and not re.fullmatch(r"[a-f0-9]{40}",args.ref):
        parser.error("Ref must be main or a full lowercase commit SHA.")
    destination=Path.home()/DIRECTORY
    if args.plan:
        print(json.dumps({"repository":REPO,"ref":args.ref,"destination":str(destination),
            "installer":"brev/install.py","existing_checkout_updates":False},indent=2))
        return
    if sys.platform != "linux" or os.geteuid()==0:
        parser.error("Run as the same non-root Linux user as Brev managed Jupyter.")
    course=prepare_checkout(destination,args.ref)
    subprocess.run([sys.executable,str(course/"brev/install.py"),"--course-dir",str(course)],check=True)

if __name__=="__main__":
    try:
        main()
    except (ValueError,OSError,subprocess.CalledProcessError) as exc:
        raise SystemExit("Workshop bootstrap stopped: "+str(exc)) from None

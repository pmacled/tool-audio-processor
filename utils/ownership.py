"""
Ownership management utilities for handling file permissions.
"""

import os


def get_workspace_owner():
    """Get the UID/GID from environment variables (HOST_UID and HOST_GID)."""
    try:
        uid = int(os.environ.get('HOST_UID', 0))
        gid = int(os.environ.get('HOST_GID', 0))
        # Only return valid non-root UID/GID
        if uid > 0 and gid > 0:
            return uid, gid
        return None, None
    except (TypeError, ValueError) as e:
        print(
            f"Warning: Invalid HOST_UID/HOST_GID environment values; "
            f"skipping ownership changes: {e}",
            flush=True,
        )
        return None, None


def fix_ownership(path):
    """Change ownership of path to match /workspace owner."""
    uid, gid = get_workspace_owner()
    if uid is not None and gid is not None:
        try:
            # If path is a directory, chown recursively
            if os.path.isdir(path):
                for root, dirs, files in os.walk(path):
                    os.chown(root, uid, gid)
                    for f in files:
                        os.chown(os.path.join(root, f), uid, gid)
            else:
                os.chown(path, uid, gid)
        except Exception as e:
            # Log but don't fail - this is a best-effort operation
            print(f"Warning: Could not change ownership of {path}: {e}", flush=True)

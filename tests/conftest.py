"""Pytest configuration for test suite.

Force a non-interactive matplotlib backend to avoid sporadic Tk/Tcl errors
(e.g., tkinter.TclError: Can't find a usable init.tcl) on local Windows runs.
CI already runs headless, but local environments may sometimes pick TkAgg.
"""

from __future__ import annotations

import os

# Set backend before any pyplot import happens.
os.environ.setdefault("MPLBACKEND", "Agg")

# (Optional future spot) Add global fixtures or marks.

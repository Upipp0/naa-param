"""Pipeline step scripts.

Each module here is both an importable step (what the orchestrator,
``naa_param/naa_param.py``, calls through its ``run()`` function) and a
standalone command-line script, so a single step can be re-run by hand while
debugging without going through ``naa-param run``.
"""

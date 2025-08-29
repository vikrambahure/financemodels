# src/equitymodels/__init__.py
try:
    from importlib.metadata import version as _pkg_version
    __version__ = _pkg_version("equitymodels")
except Exception:
    __version__ = "0.0.0"  # fallback when running from source without install

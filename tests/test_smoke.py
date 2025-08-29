import importlib

def test_package_imports():
    import equitymodels
    assert hasattr(equitymodels, "__version__")

def test_pipeline_imports():
    mod = importlib.import_module("equitymodels.pipelines")
    assert hasattr(mod, "run_analysis")

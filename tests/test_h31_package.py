from importlib import import_module


def test_simplexid_package_imports():
    modules = [
        "simplexid",
        "simplexid.data",
        "simplexid.config",
        "simplexid.parser",
        "simplexid.rules",
        "simplexid.scoring",
        "simplexid.dam",
        "simplexid.pipeline",
    ]

    for module_name in modules:
        import_module(module_name)


def test_core_symbols_import():
    from simplexid.config import VARIANTS
    from simplexid.parser import parse
    from simplexid.rules import extract_explicit_pairs
    from simplexid.pipeline import run_pipeline

    assert VARIANTS
    assert callable(parse)
    assert callable(extract_explicit_pairs)
    assert callable(run_pipeline)

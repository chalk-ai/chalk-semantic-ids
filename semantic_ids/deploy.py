"""Compatibility entrypoint for the hosting stage in host/run.py."""

if __name__ == "__main__":
    import runpy

    from semantic_ids.project import ROOT

    # The stage parses --help before its deployment decorators execute.
    runpy.run_path(str(ROOT / "host" / "run.py"), run_name="__main__")
else:
    from semantic_ids.stages import load_stage

    _host = load_stage("host")
    semantic_tokenize = _host.semantic_tokenize
    semantic_recommend = _host.semantic_recommend
    main = _host.main

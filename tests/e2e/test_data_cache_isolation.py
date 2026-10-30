"""The test suite never writes datasets into the user's real cache (~/.cache/tokenisers/data)."""

from pathlib import Path

from tokenisers.data import DataPipeline, SigmorphonLoader


def test_default_cache_dir_is_a_session_temp_dir(tmp_path_factory) -> None:
    user_cache = (Path.home() / ".cache" / "tokenisers").resolve()
    session_tmp = tmp_path_factory.getbasetemp().resolve()
    for cache_dir in (DataPipeline(force_offline=True).cache_dir, SigmorphonLoader(offline=True).cache_dir):
        resolved = cache_dir.resolve()
        assert not resolved.is_relative_to(user_cache), resolved
        assert resolved.is_relative_to(session_tmp), resolved

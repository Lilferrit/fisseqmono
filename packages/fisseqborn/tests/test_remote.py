import pathlib
import shutil
import subprocess

import polars as pl
import pytest
from conftest import PIPELINE_BATCHES

import fisseqborn as fb
from fisseqborn import _pipeline, _remote

HOST = "me@cluster"


class FakeSsh:
    """Stands in for ssh / scp: the listing runs in a local shell (so quoting and globbing
    are exercised), and scp copies local files. Every file requested is recorded."""

    def __init__(self):
        self.listings: list[str] = []
        self.copied: list[str] = []
        self.fail: str | None = None

    def __call__(self, argv):
        if argv[0] == "ssh":
            assert argv[-2] == HOST
            self.listings.append(argv[-1])
            return subprocess.run(
                ["sh", "-c", argv[-1]], capture_output=True, text=True, check=False
            )
        assert argv[0] == "scp"
        *sources, target = [a for a in argv[1:] if not a.startswith("-") and "=" not in a]
        if self.fail:
            return subprocess.CompletedProcess(argv, 1, "", self.fail)
        for src in sources:
            host, path = src.split(":", 1)
            assert host == HOST
            if not pathlib.Path(path).is_file():
                return subprocess.CompletedProcess(
                    argv, 1, "", f"scp: {path}: No such file or directory"
                )
            self.copied.append(path)
            shutil.copy(path, target)
        return subprocess.CompletedProcess(argv, 0, "", "")

    def rel(self, root) -> set[str]:
        return {str(pathlib.Path(p).relative_to(root)) for p in self.copied}


@pytest.fixture
def fake(monkeypatch):
    fake = FakeSsh()
    monkeypatch.setattr(_remote, "_run", fake)
    monkeypatch.setattr(_remote, "_TEMP_DIRS", {})
    return fake


@pytest.fixture
def remote(pipeline_dir):
    """The fixture run with large files no loader asks for, addressed as a remote spec."""
    for batch in PIPELINE_BATCHES:
        decoys = pipeline_dir / "feature_select_batchwise" / batch / "pseudo_replicates"
        decoys.mkdir()
        for i in range(3):
            pl.DataFrame({"x": [i]}).write_parquet(decoys / f"median_{i}.parquet")
    return f"{HOST}:{pipeline_dir}"


def _fs(*files):
    return {f"feature_select_batchwise/{b}/{f}" for b in PIPELINE_BATCHES for f in files}


@pytest.mark.parametrize(
    "spec, expected",
    [
        ("host:/runs/a", ("host", "/runs/a")),
        ("me@host:runs/a/", ("me@host", "runs/a")),
        ("host:~/runs", ("host", "~/runs")),
        ("/abs/path", None),
        ("rel/path", None),
        ("rel/a:b", None),
        ("host:", None),
        (pathlib.Path("host:/runs"), None),
    ],
)
def test_parse(spec, expected):
    assert _remote.parse(spec) == expected


def test_profiles_download_only_what_they_read(fake, remote, pipeline_dir, tmp_path):
    dl = tmp_path / "dl"
    profiles = fb.Profiles.from_pipeline(remote, types=["median"], download_dir=dl)
    assert fake.rel(pipeline_dir) == _fs("aggregates/median.parquet")
    assert len(fake.listings) == 1
    # files mirror the run's layout and give the same profiles as a local load
    assert (dl / "feature_select_batchwise" / "T1_R1" / "aggregates" / "median.parquet").is_file()
    assert profiles.df.equals(fb.Profiles.from_pipeline(pipeline_dir, types=["median"]).df)
    assert not list(dl.glob(".part-*"))

    fake.copied.clear()
    fb.Profiles.from_pipeline(
        remote, types=["median"], passthrough=["KSnegLogP"], metadata=True, download_dir=dl
    ).collect()
    # median is cached; only the newly requested files are copied
    assert fake.rel(pipeline_dir) == _fs(
        "passthrough_aggregates/KSnegLogP.parquet", "output.parquet"
    )


def test_batches_and_exclude_filter_before_downloading(fake, remote, pipeline_dir, tmp_path):
    fb.Profiles.from_pipeline(remote, exclude="T1*", download_dir=tmp_path / "a")
    assert fake.rel(pipeline_dir) == {"feature_select_batchwise/T2_R1/aggregates/median.parquet"}
    fake.copied.clear()
    ovwt = fb.OvwtScores.from_pipeline(
        remote, batches=["T10_R1", "T2_R1"], exclude="T2_*", download_dir=tmp_path / "b"
    )
    assert fake.rel(pipeline_dir) == {"ovwt_batchwise/T10_R1/results.parquet"}
    assert set(ovwt.df["meta_experiment"]) == {"T10_R1"}
    with pytest.raises(FileNotFoundError, match="T99"):
        fb.OvwtScores.from_pipeline(remote, batches=["T99"], download_dir=tmp_path / "c")


def test_refresh_and_temp_dir_reuse(fake, remote, pipeline_dir):
    fb.OvwtScores.from_pipeline(remote)
    first = set(_remote._TEMP_DIRS.values())
    assert len(fake.copied) == len(PIPELINE_BATCHES)
    fb.OvwtScores.from_pipeline(remote)
    assert len(fake.copied) == len(PIPELINE_BATCHES)  # reused from the session's temp dir
    assert set(_remote._TEMP_DIRS.values()) == first
    fb.OvwtScores.from_pipeline(remote, refresh=True)
    assert len(fake.copied) == 2 * len(PIPELINE_BATCHES)


def test_blocklists_list_once_and_download_only_blocklists(fake, remote, pipeline_dir, tmp_path):
    blocklists = fb.Blocklists.from_pipeline(remote, download_dir=tmp_path)
    assert fake.rel(pipeline_dir) == _fs("blocklists/median.parquet", "blocklists/KS.parquet")
    assert len(fake.listings) == 2  # the batches, then every batch's blocklists at once
    local = fb.Blocklists.from_pipeline(pipeline_dir)
    assert blocklists.df.sort(pl.all()).equals(local.df.sort(pl.all()))


def test_write_global_from_remote(fake, remote, pipeline_dir, tmp_path):
    written = fb.write_global(remote, tmp_path / "out", operations=(), download_dir=tmp_path / "dl")
    assert fake.rel(pipeline_dir) == _fs(
        "aggregates/median.parquet", "blocklists/median.parquet"
    ) | {f"ovwt_batchwise/{b}/results.parquet" for b in PIPELINE_BATCHES}
    local = fb.write_global(pipeline_dir, tmp_path / "local", operations=())
    for name, path in written.items():
        assert pl.read_parquet(path).equals(pl.read_parquet(local[name]))


def test_cli_download_dir(fake, remote, pipeline_dir, tmp_path):
    from fisseqborn import global_aggregate

    global_aggregate.main(
        [
            remote,
            "--out",
            str(tmp_path / "out"),
            "--operations",
            "--download-dir",
            str(tmp_path / "dl"),
        ]
    )
    assert (tmp_path / "dl" / "ovwt_batchwise" / "T1_R1" / "results.parquet").is_file()


def test_errors(fake, remote, pipeline_dir, tmp_path):
    dl = tmp_path / "dl"
    with pytest.raises(FileNotFoundError, match="KS2"):
        fb.Profiles.from_pipeline(remote, types=["KS2"], download_dir=dl)
    fake.fail = "Permission denied"
    with pytest.raises(RuntimeError, match="Permission denied"):
        fb.OvwtScores.from_pipeline(remote, download_dir=dl)
    assert not list(dl.rglob("*.parquet"))  # a failed copy leaves nothing behind
    with pytest.raises(FileNotFoundError, match="batch directories"):
        fb.Profiles.from_pipeline(f"{HOST}:{tmp_path / 'nope'}", download_dir=dl)
    with pytest.raises(ValueError, match="download_dir"):
        fb.Profiles.from_pipeline(pipeline_dir, download_dir=dl)


def test_dataset_read_remote_file(fake, pipeline_dir, tmp_path):
    path = pipeline_dir / "ovwt_batchwise" / "T1_R1" / "results.parquet"
    ds = fb.Dataset.read(f"{HOST}:{path}", download_dir=tmp_path)
    assert ds.df.equals(pl.read_parquet(path))
    assert (tmp_path / "results.parquet").is_file()
    with pytest.raises(ValueError, match="glob"):
        fb.Dataset.read(f"{HOST}:{path.parent}/*.parquet")


def test_source_is_shared(pipeline_dir):
    src = _pipeline.source(pipeline_dir)
    assert _pipeline.source(src) is src
    with pytest.raises(ValueError, match="Source"):
        _pipeline.source(src, download_dir="x")

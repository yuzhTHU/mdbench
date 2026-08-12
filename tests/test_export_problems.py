from pathlib import Path

from src.export_problems import get_parser, main


def test_export_bundled_problems(tmp_path):
    output_dir = tmp_path / "problems"
    args = get_parser().parse_args(["--output-dir", str(output_dir)])

    assert main(args) == 0
    exported = sorted(output_dir.glob("*.yaml"))
    bundled = sorted(Path("problems").glob("*.yaml"))
    assert [path.name for path in exported] == [path.name for path in bundled]
    assert all(path.read_bytes() == (Path("problems") / path.name).read_bytes() for path in exported)


def test_export_refuses_overwrite_without_confirmation(tmp_path, monkeypatch):
    output_dir = tmp_path / "problems"
    output_dir.mkdir()
    existing = output_dir / "demo_problem.yaml"
    existing.write_text("keep me", encoding="utf-8")
    monkeypatch.setattr("builtins.input", lambda _: "n")
    args = get_parser().parse_args(["--output-dir", str(output_dir)])

    assert main(args) == 1
    assert existing.read_text(encoding="utf-8") == "keep me"


def test_export_force_overwrites_existing_files(tmp_path):
    output_dir = tmp_path / "problems"
    output_dir.mkdir()
    existing = output_dir / "demo_problem.yaml"
    existing.write_text("old", encoding="utf-8")
    args = get_parser().parse_args(
        ["--output-dir", str(output_dir), "--force"]
    )

    assert main(args) == 0
    assert existing.read_bytes() == Path("problems/demo_problem.yaml").read_bytes()

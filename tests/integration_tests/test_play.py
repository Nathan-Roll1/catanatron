import os
import tempfile

from click.testing import CliRunner

from catanatron.cli.play import simulate


def test_play():
    runner = CliRunner()
    result = runner.invoke(simulate, ["--num=5", "--players=R,F,VP,W"])
    assert result.exit_code == 0
    assert "Game Summary" in result.output


def test_play_strong():
    runner = CliRunner()
    result = runner.invoke(simulate, ["--num=1", "--players=AB,SAB,M:2:True,G:2"])
    assert result.exit_code == 0
    assert "Game Summary" in result.output


def test_play_with_random_number_placement():
    runner = CliRunner()
    result = runner.invoke(
        simulate,
        [
            "--num=1",
            "--players=R,R",
            "--config-number-placement=random",
        ],
    )
    assert result.exit_code == 0
    assert "Game Summary" in result.output


def test_play_with_friendly_robber():
    runner = CliRunner()
    result = runner.invoke(
        simulate,
        [
            "--num=1",
            "--players=R,R",
            "--config-friendly-robber",
        ],
    )
    assert result.exit_code == 0
    assert "Game Summary" in result.output


def test_play_rejects_official_spiral_for_tournament():
    runner = CliRunner()
    result = runner.invoke(
        simulate,
        [
            "--num=1",
            "--players=R,R",
            "--config-map=TOURNAMENT",
            "--config-number-placement=official_spiral",
        ],
    )
    assert result.exit_code != 0
    assert result.exception is not None
    assert "official_spiral number placement is only supported for" in str(
        result.exception
    )


def test_json_output():
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdirname:
        result = runner.invoke(
            simulate,
            [
                "--num=2",
                "--players=R,R",
                "--output",
                tmpdirname,
            ],
        )
        assert result.exit_code == 0
        files = os.listdir(tmpdirname)
        assert len(files) == 2
        assert all(f.endswith(".json") for f in files)

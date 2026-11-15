import subprocess

import pytest

from app import provenance
from app.provenance import COMMIT_ENV, git_commit


def fake_git(head: str, status: str):
    return lambda *args: {'rev-parse': head, 'status': status}[args[0]]


def test_image_commit_wins(monkeypatch):
    monkeypatch.setenv(COMMIT_ENV, 'abc1234')
    monkeypatch.setattr(provenance, '_git', lambda *args: pytest.fail('git run although SCREENER_COMMIT is set'))
    assert git_commit() == 'abc1234'


def test_image_built_without_a_revision_has_no_commit(monkeypatch):
    monkeypatch.setenv(COMMIT_ENV, 'unknown')
    assert git_commit() is None


def test_clean_checkout_gives_its_head(monkeypatch):
    monkeypatch.delenv(COMMIT_ENV, raising=False)
    monkeypatch.setattr(provenance, '_git', fake_git('a' * 40, ''))
    assert git_commit() == 'a' * 40


def test_edited_checkout_is_marked_dirty(monkeypatch):
    monkeypatch.delenv(COMMIT_ENV, raising=False)
    monkeypatch.setattr(provenance, '_git', fake_git('a' * 40, ' M app/analyzer.py'))
    assert git_commit() == 'a' * 40 + '-dirty'


@pytest.mark.parametrize('error', [OSError('no git'), subprocess.CalledProcessError(128, 'git')])
def test_no_git_gives_no_commit(monkeypatch, error):
    monkeypatch.delenv(COMMIT_ENV, raising=False)

    def fail(*args):
        raise error

    monkeypatch.setattr(provenance, '_git', fail)
    assert git_commit() is None


def test_this_checkout_reports_a_commit(monkeypatch):
    monkeypatch.delenv(COMMIT_ENV, raising=False)
    commit = git_commit()
    assert commit is not None and len(commit.removesuffix('-dirty')) == 40

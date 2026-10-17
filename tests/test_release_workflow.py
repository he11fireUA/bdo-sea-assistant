from pathlib import Path
import re


WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "release.yml"
BUILD_WINDOWS = Path(__file__).resolve().parents[1] / "scripts" / "build_windows.ps1"


def test_release_workflow_builds_on_review_but_publishes_only_tags() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "pull_request:" in text
    assert 'tags:\n      - "v*.*.*"' in text
    assert "contents: read" in text
    assert re.search(
        r"publish:\s+if: startsWith\(github\.ref, 'refs/tags/v'\).*?permissions:\s+contents: write",
        text,
        re.DOTALL,
    )
    assert "needs: build" in text


def test_write_privileged_release_values_are_passed_through_environment() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "RELEASE_TAG: ${{ github.ref_name }}" in text
    assert "EXPECTED_SHA: ${{ github.sha }}" in text
    assert "REPOSITORY: ${{ github.repository }}" in text
    assert '$tag = "${{ github.ref_name }}"' not in text
    assert '"${{ github.repository }}"' not in text


def test_publish_rechecks_the_remote_tag_and_cleans_only_its_own_failed_draft() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "git ls-remote --tags" in text
    assert 'if ($resolvedSha -ne $env:EXPECTED_SHA)' in text
    assert "function Get-GitHubReleaseByTag" in text
    assert "function Get-GitHubReleaseById" in text
    assert 'releases?per_page=100&page=$page' in text
    assert "$pageItems.Count -lt 100" in text
    assert "Multiple releases exist for tag" in text
    assert "/releases/tags/" not in text
    assert "StatusCode.value__ -eq 404" in text
    assert 'Invoke-RestMethod -Method Post -Uri "https://api.github.com/repos/$($env:REPOSITORY)/releases"' in text
    assert 'tag_name = $tag' in text
    assert 'target_commitish = $env:EXPECTED_SHA' in text
    assert "gh release create" not in text
    assert "$createdReleaseId = [long]$createdRelease.id" in text
    assert 'if ($createdReleaseId -le 0)' in text
    assert "if ($draftCreated -and $createdReleaseId)" in text
    assert 'gh api --method DELETE "repos/$($env:REPOSITORY)/releases/$createdReleaseId"' in text
    assert "$remaining = Get-GitHubReleaseById $createdReleaseId" in text
    assert "Draft cleanup verification failed" in text
    assert "gh release delete" not in text


def test_release_workflow_checks_native_commands_and_main_provenance() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "uv sync failed with exit code" in text
    assert text.count("git fetch origin main --no-tags") >= 2
    assert "git merge-base --is-ancestor $env:EXPECTED_SHA origin/main" in text
    assert text.count("Confirm-MainContainment") >= 3
    assert "Failed to fetch origin/main" in text
    assert "Verified release commit is no longer contained in origin/main" in text


def test_windows_build_script_fails_immediately_on_native_command_errors() -> None:
    text = BUILD_WINDOWS.read_text(encoding="utf-8")

    assert "$PSNativeCommandUseErrorActionPreference = $true" in text
    for message in (
        "uv sync failed",
        "Test suite failed",
        "PyInstaller failed",
    ):
        assert message in text


def test_release_workflow_pins_every_third_party_action_to_a_commit() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    action_refs = re.findall(r"uses:\s*([^\s#]+)", text)

    assert action_refs
    assert all(re.fullmatch(r"[^@]+@[0-9a-f]{40}", ref) for ref in action_refs)

"""Keep private production assets opt-in without weakening their validation."""

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--production-assets",
        action="store_true",
        default=False,
        help="Run integration checks against the original local frozen releases/media.",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--production-assets"):
        return
    skip = pytest.mark.skip(
        reason="Requires original production assets; opt in with --production-assets."
    )
    for item in items:
        if item.get_closest_marker("production_assets"):
            item.add_marker(skip)

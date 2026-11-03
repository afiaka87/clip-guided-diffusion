"""Downloads, from a real HTTP server on this machine."""

import hashlib
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import NamedTuple

import pytest
import requests

from cgd import download

BODY = bytes(range(256)) * 1024  # Several chunks.


class Server(NamedTuple):
    """A server running in this process."""

    url: str
    requests: list[str]  # The path of each request, in order.


def handler(requested: list[str]) -> type[BaseHTTPRequestHandler]:
    """Serves BODY at /file, a response shorter than it claims at /short, and 404 elsewhere."""

    class Handler(BaseHTTPRequestHandler):
        """Answers GET requests as handler() says."""

        def do_GET(self) -> None:  # pylint: disable=invalid-name  # The name http.server calls.
            """Answer one request, and note its path."""
            requested.append(self.path)
            if self.path not in {"/file", "/short"}:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Length", str(len(BODY) + (self.path == "/short")))
            self.end_headers()
            self.wfile.write(BODY)

        def log_message(self, *_: object) -> None:
            """Keep the test output clean."""

    return Handler


@pytest.fixture(name="server")
def fixture_server() -> Iterator[Server]:
    """A server on a free port of this machine, for the length of one test."""
    requested: list[str] = []
    with ThreadingHTTPServer(("127.0.0.1", 0), handler(requested)) as http_server:
        thread = threading.Thread(target=http_server.serve_forever, daemon=True)
        thread.start()
        try:
            yield Server(f"http://127.0.0.1:{http_server.server_address[1]}", requested)
        finally:
            http_server.shutdown()
            thread.join()


@pytest.fixture
def one_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail at once instead of waiting through the retries."""
    monkeypatch.setattr(download, "ATTEMPTS", 1)


class TestDownload:
    def test_it_saves_the_file_and_returns_its_path(self, server: Server, tmp_path: Path) -> None:
        path = download.download(f"{server.url}/file", "weights.pt", tmp_path / "new" / "folder")
        assert path == tmp_path / "new" / "folder" / "weights.pt"
        assert path.read_bytes() == BODY

    def test_a_cached_file_is_not_downloaded_again(self, server: Server, tmp_path: Path) -> None:
        first = download.download(f"{server.url}/file", "weights.pt", tmp_path)
        assert download.download(f"{server.url}/file", "weights.pt", tmp_path) == first
        assert server.requests == ["/file"]

    @pytest.mark.usefixtures("one_attempt")
    def test_an_incomplete_download_fails_and_leaves_nothing(self, server: Server, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="Download failed after 1 attempts"):
            download.download(f"{server.url}/short", "weights.pt", tmp_path)
        assert not any(tmp_path.iterdir())

    @pytest.mark.usefixtures("one_attempt")
    def test_an_http_error_fails(self, server: Server, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="404"):
            download.download(f"{server.url}/missing", "weights.pt", tmp_path)

    def test_a_folder_in_the_way_is_explained(self, tmp_path: Path) -> None:
        (tmp_path / "weights.pt").mkdir()
        with pytest.raises(RuntimeError, match="is not a regular file"):
            download.download("http://127.0.0.1:9/file", "weights.pt", tmp_path)

    def test_a_verified_download_checks_the_digest(self, server: Server, tmp_path: Path) -> None:
        digest = hashlib.sha256(BODY).hexdigest()
        assert download.download_verified(f"{server.url}/file", "weights.pt", digest, tmp_path).read_bytes() == BODY
        with pytest.raises(RuntimeError, match="Delete it to download it again"):
            download.download_verified(f"{server.url}/file", "weights.pt", "0" * 64, tmp_path)


class TestRetry:
    def test_it_returns_the_first_success(self) -> None:
        outcomes = iter([RuntimeError("one"), requests.ConnectionError("two"), "done"])
        waits: list[float] = []

        def action() -> str:
            outcome = next(outcomes)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        assert download.retry(action, 3, waits.append) == "done"
        assert waits == [1, 2]

    def test_the_last_failure_raises(self) -> None:
        waits: list[float] = []

        def action() -> str:
            raise RuntimeError(f"attempt {len(waits) + 1}")

        with pytest.raises(RuntimeError, match="attempt 3"):
            download.retry(action, 3, waits.append)
        assert waits == [1, 2]

    def test_other_errors_are_not_retried(self) -> None:
        waits: list[float] = []

        def action() -> str:
            raise ValueError("a bug, not a bad connection")

        with pytest.raises(ValueError, match="a bug"):
            download.retry(action, 3, waits.append)
        assert not waits


class TestFetch:
    def test_a_file(self, tmp_path: Path) -> None:
        path = tmp_path / "image.png"
        path.write_bytes(BODY)
        assert download.fetch(path).read() == BODY

    def test_a_url(self, server: Server) -> None:
        assert download.fetch(f"{server.url}/file").read() == BODY

    @pytest.mark.parametrize(
        ("path", "expected"),
        [("https://a.b/c.png", True), ("http://a.b/c.png", True), ("images/c.png", False), ("ftp://a.b/c", False)],
    )
    def test_is_url(self, path: str, expected: bool) -> None:
        assert download.is_url(path) is expected

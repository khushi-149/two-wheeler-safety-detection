"""
Tiny local MJPEG server.

Streamlit can only refresh an image a few times per second, which looks like a
slideshow. This server streams frames to the browser as a continuous video
(multipart JPEG), so playback is smooth no matter how slow detection is.

It listens on 127.0.0.1 only: the page must be opened in a browser on the same
computer that runs the dashboard.
"""

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

BOUNDARY = "frame"


class _Handler(BaseHTTPRequestHandler):

    def log_message(self, *args):  # keep the terminal quiet
        pass

    def do_GET(self):
        hub = self.server.hub
        path = self.path.split("?")[0]

        if path == "/stream":
            self._stream(hub)
        elif path == "/latest.jpg":
            jpeg, _ = hub.latest()

            if jpeg is None:
                self.send_error(404, "No frame yet")
                return

            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(jpeg)))
            self.end_headers()
            self.wfile.write(jpeg)
        else:
            self.send_error(404)

    def _stream(self, hub):
        self.send_response(200)
        self.send_header("Cache-Control", "no-cache, no-store, private")
        self.send_header("Pragma", "no-cache")
        self.send_header(
            "Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY}"
        )
        self.end_headers()

        last_seq = -1

        try:
            while not hub.closed:
                jpeg, seq = hub.wait_for_frame(last_seq, timeout=2.0)

                if jpeg is None:
                    continue

                # If nothing new arrived within the timeout the same frame is
                # sent again: it keeps the connection alive and lets us notice
                # when the browser has gone away.
                last_seq = seq

                self.wfile.write(
                    f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\n"
                    f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                )
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError,
                ConnectionAbortedError, OSError):
            pass  # browser closed the stream


class MJPEGServer:

    def __init__(self, host="127.0.0.1", port=8765, max_tries=25, quality=80):
        self.quality = quality
        self.closed = False
        self._jpeg = None
        self._seq = 0
        self._cond = threading.Condition()

        last_error = None
        self.httpd = None

        for candidate in range(port, port + max_tries):
            try:
                self.httpd = ThreadingHTTPServer((host, candidate), _Handler)
                self.port = candidate
                break
            except OSError as error:
                last_error = error

        if self.httpd is None:
            raise RuntimeError(f"No free port for the live stream: {last_error}")

        self.httpd.daemon_threads = True
        self.httpd.hub = self

        self._thread = threading.Thread(
            target=self.httpd.serve_forever, daemon=True
        )
        self._thread.start()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}/stream"

    def push(self, frame_bgr):
        """Publish a new frame (any thread)."""
        ok, buffer = cv2.imencode(
            ".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, self.quality]
        )

        if not ok:
            return

        with self._cond:
            self._jpeg = buffer.tobytes()
            self._seq += 1
            self._cond.notify_all()

    def latest(self):
        with self._cond:
            return self._jpeg, self._seq

    def wait_for_frame(self, last_seq, timeout=2.0):
        with self._cond:
            self._cond.wait_for(
                lambda: self._seq != last_seq or self.closed, timeout
            )
            return self._jpeg, self._seq

    def close(self):
        """Stop the server. Safe to call more than once."""
        if self.closed:
            return

        self.closed = True

        with self._cond:
            self._cond.notify_all()

        try:
            self.httpd.shutdown()
            self.httpd.server_close()
        except OSError:
            pass  # already closed
import numpy as np

from display import OverlayDisplay

SIZE = (40, 30)          # (width, height) of the preview
FPS = 25


class FakeCap:
    def __init__(self):
        self.frame, self.seq = None, 0

    def peek(self):
        return self.frame, self.seq

    def deliver(self, seq):
        self.seq = seq
        self.frame = np.full((30, 40, 3), seq % 250, np.uint8)      # frame colour = its number


def overlay(color):
    mask = np.zeros((30, 40, 1), bool)
    mask[10:20, 10:20] = True
    return mask, np.full((30, 40, 3), color, np.uint8)


def make(aligned=True, **kw):
    cap, shown = FakeCap(), []
    display = OverlayDisplay(cap, shown.append, SIZE, FPS, aligned=aligned,
                             min_delay=0.4, max_delay=2.0, **kw)
    return cap, display, shown


def play(cap, display, frames, start=0.0):
    """Deliver frames at 25 fps and step the display every 10 ms; returns the last time."""
    now = start
    for seq in range(1, frames + 1):
        cap.deliver(seq)
        for _ in range(4):
            display.step(now)
            now += 0.01
    return now


def test_each_frame_is_shown_once_in_order_after_the_delay():
    cap, display, shown = make()
    play(cap, display, 40)

    colours = [int(img[0, 0, 0]) for img in shown]
    assert colours == sorted(set(colours))                   # in order, no repeats
    assert colours[0] <= 3 and len(colours) >= 20            # started after ~0.4 s, kept pace


def test_frame_gets_the_detections_made_on_that_frame_not_the_newest_ones():
    """Regression: boxes used to come from a frame ~10 frames newer than the picture."""
    cap, display, shown = make()

    display.push_overlay(5, *overlay(111))
    display.push_overlay(25, *overlay(222))                   # much newer detection
    play(cap, display, 50)

    by_frame = {int(img[0, 0, 0]): img for img in shown}
    assert by_frame[5][15, 15, 0] == 111                      # frame 5 -> detection of frame 5
    assert by_frame[8][15, 15, 0] == 111                      # nearest earlier detection
    assert by_frame[22][15, 15, 0] == 222                     # nearest is now the later one


def test_no_overlay_when_the_nearest_detection_is_too_old():
    cap, display, shown = make()
    display.push_overlay(2, *overlay(111))
    play(cap, display, 60)

    by_frame = {int(img[0, 0, 0]): img for img in shown}
    assert by_frame[2][15, 15, 0] == 111
    assert by_frame[50][15, 15, 0] == 50                      # stale detection is not drawn


def test_delay_adapts_to_detection_time_with_limits_and_gentle_changes():
    cap, display, _ = make()
    assert display.target_delay == 0.4

    for _ in range(30):
        display.report_latency(0.5)
    assert 0.75 < display.target_delay < 0.85                 # 1.3 * 0.5 + 0.15

    for _ in range(100):
        display.report_latency(30.0)
    assert display.target_delay == 2.0                        # capped

    display.step(0.0)
    display.step(1.0)
    assert display.delay <= 0.4 + 0.15 * 1.0 + 1e-9           # eased, not jumped


def test_live_mode_shows_newest_frame_and_latest_detection():
    cap, display, shown = make(aligned=False)
    display.push_overlay(1, *overlay(111))
    display.push_overlay(9, *overlay(222))

    cap.deliver(30)
    display.step(0.0)

    assert int(shown[0][0, 0, 0]) == 30 and shown[0][15, 15, 0] == 222


def test_history_is_bounded():
    cap, display, _ = make()
    play(cap, display, 400)
    assert len(display.history) <= int((2.0 + 0.5) * FPS) + 2
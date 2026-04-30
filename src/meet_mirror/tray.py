from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

from loguru import logger
from PIL import Image, ImageDraw
from pystray import Icon, Menu, MenuItem

from .pipeline import Pipeline


def _make_icon(color: tuple[int, int, int]) -> Image.Image:
    size = 64
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    pad = 4
    d.ellipse((pad, pad, size - pad, size - pad), fill=color + (255,))
    return img


_ICON_RUNNING = _make_icon((40, 200, 80))   # green
_ICON_IDLE = _make_icon((140, 140, 140))    # gray


def _open_folder(path: Path) -> None:
    try:
        if sys.platform == "win32":
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except Exception as e:
        logger.warning(f"Failed to open {path}: {e}")


class TrayApp:
    """System tray icon + global hotkey for pipeline toggle.

    Runs the tray's message loop in a background thread (run_detached) so
    Qt keeps the foreground main loop. Pipeline start/stop are guarded by
    Pipeline's internal lock.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        hotkey: str,
        on_quit: Callable[[], None],
    ) -> None:
        self.pipeline = pipeline
        self.hotkey = hotkey
        self.on_quit = on_quit
        self._icon: Icon | None = None
        self._hotkey_handle = None
        self._heartbeat_stop = threading.Event()
        self._heartbeat: threading.Thread | None = None

    # --- menu actions ---

    def _toggle(self, _icon=None, _item=None) -> None:
        try:
            if self.pipeline.is_running:
                self.pipeline.stop()
            else:
                # Pipeline.start spawns workers and loads models — can take
                # ~10s. Run on a thread so the tray menu stays responsive.
                threading.Thread(
                    target=self.pipeline.start,
                    name="PipelineStarter",
                    daemon=True,
                ).start()
            self._refresh_icon()
        except Exception as e:
            logger.exception(f"Toggle failed: {e}")

    def _restart(self, _icon=None, _item=None) -> None:
        try:
            self.pipeline.stop()
            threading.Thread(
                target=self.pipeline.start, name="PipelineStarter", daemon=True
            ).start()
            self._refresh_icon()
        except Exception as e:
            logger.exception(f"Restart failed: {e}")

    def _open_session(self, _icon=None, _item=None) -> None:
        d = self.pipeline.session_dir
        if d is None:
            logger.info("No session yet")
            return
        _open_folder(d)

    def _quit(self, _icon=None, _item=None) -> None:
        self.stop()
        self.on_quit()

    # --- icon state ---

    def _refresh_icon(self) -> None:
        if self._icon is None:
            logger.warning("_refresh_icon called but self._icon is None")
            return
        running = self.pipeline.is_running
        try:
            if running:
                self._icon.icon = _ICON_RUNNING
                self._icon.title = "Meet Mirror (running)"
            else:
                self._icon.icon = _ICON_IDLE
                self._icon.title = "Meet Mirror (idle)"
            self._icon.update_menu()
            logger.info(f"Tray icon refreshed: running={running}")
        except Exception as e:
            logger.exception(f"Tray icon refresh failed: {e}")

    def _heartbeat_loop(self) -> None:
        """Periodically reconcile the icon with the pipeline state.

        Pipeline.start runs on a worker thread (so the tray menu stays
        responsive during the ~10s ML model load), so we cannot refresh
        the icon synchronously after _toggle() returns. This loop catches
        the state change as soon as the worker thread finishes loading.
        """
        logger.info("Tray heartbeat started")
        last = None
        while not self._heartbeat_stop.is_set():
            try:
                cur = self.pipeline.is_running
                if cur != last:
                    logger.info(
                        f"Tray heartbeat: state change {last} -> {cur}"
                    )
                    self._refresh_icon()
                    last = cur
            except Exception as e:
                logger.exception(f"Tray heartbeat loop error: {e}")
            time.sleep(0.4)
        logger.info("Tray heartbeat stopped")

    # --- menu factory ---

    def _build_menu(self) -> Menu:
        return Menu(
            MenuItem(
                lambda _i: "Stop" if self.pipeline.is_running else "Start",
                self._toggle,
                default=True,
            ),
            MenuItem(
                "Open current session",
                self._open_session,
                enabled=lambda _i: self.pipeline.session_dir is not None,
            ),
            MenuItem("Restart", self._restart),
            Menu.SEPARATOR,
            MenuItem("Quit", self._quit),
        )

    # --- lifecycle ---

    def start(self) -> None:
        self._icon = Icon(
            name="meet-mirror",
            icon=_ICON_IDLE,
            title="Meet Mirror (idle)",
            menu=self._build_menu(),
        )
        self._icon.run_detached()
        logger.info("Tray icon running")

        self._heartbeat_stop.clear()
        self._heartbeat = threading.Thread(
            target=self._heartbeat_loop, name="TrayHeartbeat", daemon=True
        )
        self._heartbeat.start()

        # Global hotkey via the keyboard package. Best-effort: degrade
        # cleanly if the package can't install hooks (e.g. non-admin
        # security software blocks it).
        try:
            import keyboard

            self._hotkey_handle = keyboard.add_hotkey(self.hotkey, self._toggle)
            logger.info(f"Global hotkey registered: {self.hotkey}")
        except Exception as e:
            logger.warning(
                f"Could not register global hotkey '{self.hotkey}': {e}. "
                "Use the tray menu to toggle the pipeline."
            )

    def stop(self) -> None:
        self._heartbeat_stop.set()
        if self._heartbeat is not None:
            self._heartbeat.join(timeout=1.0)
            self._heartbeat = None
        if self._hotkey_handle is not None:
            try:
                import keyboard

                keyboard.remove_hotkey(self._hotkey_handle)
            except Exception:
                pass
            self._hotkey_handle = None
        if self._icon is not None:
            try:
                self._icon.stop()
            except Exception:
                pass
            self._icon = None

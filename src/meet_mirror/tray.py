from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from loguru import logger
from PyQt6.QtCore import QSize, Qt, QTimer
from PyQt6.QtGui import QAction, QColor, QIcon, QPainter, QPixmap
from PyQt6.QtWidgets import QMenu, QSystemTrayIcon

from .pipeline import Pipeline

_RUNNING = QColor(40, 200, 80)
_IDLE = QColor(140, 140, 140)


def _make_icon(color: QColor) -> QIcon:
    px = QPixmap(QSize(64, 64))
    px.fill(Qt.GlobalColor.transparent)
    painter = QPainter(px)
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(color)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(4, 4, 56, 56)
    finally:
        painter.end()
    return QIcon(px)


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

    Backed by Qt's QSystemTrayIcon so all icon mutations happen on the
    Qt main thread, avoiding the cross-thread NIM_MODIFY issues we hit
    with pystray on Windows. A QTimer reconciles the icon with
    Pipeline.is_running every 400 ms.
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
        self._tray: QSystemTrayIcon | None = None
        self._action_toggle: QAction | None = None
        self._action_open: QAction | None = None
        self._heartbeat: QTimer | None = None
        self._last_state: bool | None = None
        self._hotkey_handle = None
        self._starter_thread = None

    # --- menu actions (run on Qt main thread) ---

    def _toggle(self) -> None:
        """Spawn the start/stop call on a worker thread.

        Called from both the Qt main thread (menu) and the keyboard hook
        thread (global hotkey), so we must not touch Qt objects here.
        The heartbeat QTimer (main thread) reconciles the icon afterwards.
        """
        import threading

        target = self.pipeline.stop if self.pipeline.is_running else self.pipeline.start
        try:
            self._starter_thread = threading.Thread(
                target=target, name="PipelineToggle", daemon=True
            )
            self._starter_thread.start()
        except Exception:
            logger.exception("Toggle failed")

    def _restart(self) -> None:
        """Stop then start, both on a worker thread."""
        import threading

        def _do() -> None:
            try:
                self.pipeline.stop()
                self.pipeline.start()
            except Exception:
                logger.exception("Restart failed")

        threading.Thread(target=_do, name="PipelineRestart", daemon=True).start()

    def _open_session(self) -> None:
        d = self.pipeline.session_dir
        if d is None:
            logger.info("No session yet")
            return
        _open_folder(d)

    def _quit(self) -> None:
        self.stop()
        self.on_quit()

    # --- icon + menu state ---

    def _refresh_icon(self) -> None:
        if self._tray is None:
            return
        running = self.pipeline.is_running
        color = _RUNNING if running else _IDLE
        self._tray.setIcon(_make_icon(color))
        self._tray.setToolTip(
            "Meet Mirror (running)" if running else "Meet Mirror (idle)"
        )
        if self._action_toggle is not None:
            self._action_toggle.setText("Stop" if running else "Start")
        if self._action_open is not None:
            self._action_open.setEnabled(self.pipeline.session_dir is not None)

    def _heartbeat_tick(self) -> None:
        cur = self.pipeline.is_running
        if cur != self._last_state:
            logger.info(f"Tray state change {self._last_state} -> {cur}")
            self._refresh_icon()
            self._last_state = cur

    # --- lifecycle ---

    def start(self) -> None:
        if not QSystemTrayIcon.isSystemTrayAvailable():
            logger.warning(
                "System tray not available on this platform; "
                "use the global hotkey to toggle the pipeline."
            )

        self._tray = QSystemTrayIcon()
        self._tray.setIcon(_make_icon(_IDLE))
        self._tray.setToolTip("Meet Mirror (idle)")

        menu = QMenu()
        self._action_toggle = QAction("Start")
        self._action_toggle.triggered.connect(self._toggle)
        menu.addAction(self._action_toggle)

        self._action_open = QAction("Open current session")
        self._action_open.triggered.connect(self._open_session)
        self._action_open.setEnabled(False)
        menu.addAction(self._action_open)

        action_restart = QAction("Restart")
        action_restart.triggered.connect(self._restart)
        menu.addAction(action_restart)

        menu.addSeparator()

        action_quit = QAction("Quit")
        action_quit.triggered.connect(self._quit)
        menu.addAction(action_quit)

        self._tray.setContextMenu(menu)
        # Default left-click on the icon also toggles
        self._tray.activated.connect(
            lambda reason: self._toggle()
            if reason == QSystemTrayIcon.ActivationReason.Trigger
            else None
        )
        self._tray.show()
        logger.info("Tray icon visible")

        self._last_state = False
        self._heartbeat = QTimer()
        self._heartbeat.setInterval(400)
        self._heartbeat.timeout.connect(self._heartbeat_tick)
        self._heartbeat.start()

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
        if self._heartbeat is not None:
            self._heartbeat.stop()
            self._heartbeat = None
        if self._hotkey_handle is not None:
            try:
                import keyboard

                keyboard.remove_hotkey(self._hotkey_handle)
            except Exception:
                pass
            self._hotkey_handle = None
        if self._tray is not None:
            self._tray.hide()
            self._tray = None

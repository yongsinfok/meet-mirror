from __future__ import annotations

import queue
from pathlib import Path

from loguru import logger
from PyQt6.QtCore import (
    QEasingCurve,
    QPoint,
    QPropertyAnimation,
    QRectF,
    Qt,
    QTimer,
    pyqtProperty,
)
from PyQt6.QtGui import (
    QColor,
    QFont,
    QFontMetrics,
    QPainter,
    QPainterPath,
    QPen,
)
from PyQt6.QtWidgets import QApplication, QWidget

from .config import Config
from .types import ZhSegment


def _has_cjk(s: str) -> bool:
    return any("一" <= c <= "鿿" for c in s)


class SubtitleWindow(QWidget):
    """Floating, frameless, always-on-top subtitle bar.

    Default state: click-through (WA_TransparentForMouseEvents). To
    reposition, launch with --unlock (drag mode) and double-click after
    moving to lock + persist position. Slice 5 will add a tray menu /
    global hotkey to toggle drag mode at runtime.
    """

    BAR_PADDING_X = 24
    BAR_PADDING_Y = 14
    BAR_CORNER = 12

    def __init__(
        self,
        subtitle_q: queue.Queue[ZhSegment],
        config: Config,
        config_path: str | Path = "config.yaml",
        unlock: bool = False,
    ) -> None:
        super().__init__()
        self.subtitle_q = subtitle_q
        self.config = config
        self.config_path = Path(config_path)
        self.cfg = config.subtitle
        self._current_zh: str = ""
        self._current_en: str = ""
        self._opacity_value: float = 0.0
        self._drag_mode: bool = unlock
        self._drag_offset: QPoint | None = None

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents,
            not self._drag_mode,
        )

        self._init_geometry()

        self._fade_in = QPropertyAnimation(self, b"fade_opacity")
        self._fade_in.setDuration(200)
        self._fade_in.setStartValue(0.0)
        self._fade_in.setEndValue(1.0)
        self._fade_in.setEasingCurve(QEasingCurve.Type.OutCubic)

        self._fade_out = QPropertyAnimation(self, b"fade_opacity")
        self._fade_out.setDuration(400)
        self._fade_out.setStartValue(1.0)
        self._fade_out.setEndValue(0.0)
        self._fade_out.setEasingCurve(QEasingCurve.Type.InCubic)

        self._hold_timer = QTimer(self)
        self._hold_timer.setSingleShot(True)
        self._hold_timer.timeout.connect(self._fade_out.start)

        self._poll = QTimer(self)
        self._poll.setInterval(50)
        self._poll.timeout.connect(self._poll_queue)
        self._poll.start()
        self._first_segment_logged = False

        if self._drag_mode:
            # Show a placeholder so the user can see + grab the bar
            self._current_en = "drag mode"
            self._current_zh = "(drop & double-click to lock)"
            self._opacity_value = 1.0

    # --- opacity property for QPropertyAnimation ---

    def _get_opacity(self) -> float:
        return self._opacity_value

    def _set_opacity(self, value: float) -> None:
        self._opacity_value = max(0.0, min(1.0, value))
        self.update()

    fade_opacity = pyqtProperty(float, _get_opacity, _set_opacity)

    # --- geometry ---

    def _init_geometry(self) -> None:
        screen = QApplication.primaryScreen().availableGeometry()
        width = min(1200, int(screen.width() * 0.7))
        # Allow room for 1 EN line (small) + 2 ZH lines (full size) + padding
        height = max(self.cfg.font_size * 6, 140)
        self.resize(width, height)
        x = self.cfg.position.x
        y = self.cfg.position.y
        if x is None or y is None:
            x = screen.x() + (screen.width() - width) // 2
            y = screen.y() + screen.height() - height - 80
        self.move(x, y)

    # --- queue polling ---

    def _poll_queue(self) -> None:
        try:
            seg = self.subtitle_q.get_nowait()
        except queue.Empty:
            return
        if not self._first_segment_logged:
            logger.info(
                f"First subtitle delivered to UI: en={seg.en_text!r} zh={seg.zh_text!r}"
            )
            self._first_segment_logged = True
        else:
            logger.debug(f"UI segment: zh={seg.zh_text!r}")
        if not seg.zh_text and not seg.en_text:
            return
        self._current_en = seg.en_text or ""
        self._current_zh = seg.zh_text or ""
        self._fade_out.stop()
        self._hold_timer.stop()
        if self._opacity_value < 1.0:
            self._fade_in.start()
        else:
            self.update()
        self._hold_timer.start(self.cfg.hold_seconds * 1000)

    # --- painting ---

    def _wrap_text(
        self, text: str, metrics: QFontMetrics, max_w: int
    ) -> list[str]:
        if metrics.horizontalAdvance(text) <= max_w:
            return [text]

        if _has_cjk(text):
            line1 = ""
            for i, c in enumerate(text):
                if metrics.horizontalAdvance(line1 + c) > max_w:
                    line2 = text[i:]
                    return [line1, self._truncate(line2, metrics, max_w)]
                line1 += c
            return [line1]

        words = text.split()
        line1 = ""
        cut_idx = len(words)
        for i, w in enumerate(words):
            test = (line1 + " " + w).strip()
            if metrics.horizontalAdvance(test) > max_w:
                cut_idx = i
                break
            line1 = test
        line2 = " ".join(words[cut_idx:])
        return [line1, self._truncate(line2, metrics, max_w)]

    @staticmethod
    def _truncate(s: str, metrics: QFontMetrics, max_w: int) -> str:
        if metrics.horizontalAdvance(s) <= max_w:
            return s
        ell = "…"
        while len(s) > 1 and metrics.horizontalAdvance(s + ell) > max_w:
            s = s[:-1]
        return s + ell

    def paintEvent(self, _ev) -> None:
        if self._opacity_value <= 0.001:
            return
        if not self._current_zh and not self._current_en:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setOpacity(self._opacity_value)

        zh_font = QFont(self.cfg.font_family, self.cfg.font_size)
        zh_font.setBold(True)
        zh_metrics = QFontMetrics(zh_font)

        en_pt = max(10, int(self.cfg.font_size * 0.65))
        en_font = QFont(self.cfg.font_family, en_pt)
        en_metrics = QFontMetrics(en_font)

        max_text_w = self.width() - 2 * self.BAR_PADDING_X

        en_lines: list[str] = []
        if self._current_en:
            en_lines = self._wrap_text(self._current_en, en_metrics, max_text_w)[:1]
            if en_lines:
                en_lines[0] = self._truncate(en_lines[0], en_metrics, max_text_w)

        zh_lines: list[str] = []
        if self._current_zh:
            zh_lines = self._wrap_text(self._current_zh, zh_metrics, max_text_w)[:2]
            if len(zh_lines) == 2:
                zh_lines[1] = self._truncate(zh_lines[1], zh_metrics, max_text_w)

        en_h = en_metrics.lineSpacing() if en_lines else 0
        zh_block_h = zh_metrics.lineSpacing() * len(zh_lines)
        gap = 4 if en_lines and zh_lines else 0
        text_h = en_h + gap + zh_block_h

        bar_w = self.width()
        bar_h = text_h + 2 * self.BAR_PADDING_Y
        bar_x = 0.0
        bar_y = (self.height() - bar_h) / 2
        rect = QRectF(bar_x, bar_y, bar_w, bar_h)

        bg_color = QColor(self.cfg.background_color)
        bg_color.setAlphaF(self.cfg.background_opacity)
        path = QPainterPath()
        path.addRoundedRect(rect, self.BAR_CORNER, self.BAR_CORNER)
        painter.fillPath(path, bg_color)

        if self._drag_mode:
            painter.setPen(QPen(QColor(255, 200, 0), 2))
            painter.drawPath(path)

        zh_color = QColor(self.cfg.font_color)
        en_color = QColor("#CCCCCC")
        outline_color = QColor("#000000")

        cursor_y = bar_y + self.BAR_PADDING_Y

        if en_lines:
            painter.setFont(en_font)
            line = en_lines[0]
            tw = en_metrics.horizontalAdvance(line)
            tx = (bar_w - tw) / 2
            ty = cursor_y + en_metrics.lineSpacing() - en_metrics.descent()
            painter.setPen(outline_color)
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                painter.drawText(int(tx + dx), int(ty + dy), line)
            painter.setPen(en_color)
            painter.drawText(int(tx), int(ty), line)
            cursor_y += en_metrics.lineSpacing() + gap

        if zh_lines:
            painter.setFont(zh_font)
            line_h = zh_metrics.lineSpacing()
            for i, line in enumerate(zh_lines):
                tw = zh_metrics.horizontalAdvance(line)
                tx = (bar_w - tw) / 2
                ty = cursor_y + (i + 1) * line_h - zh_metrics.descent()
                painter.setPen(outline_color)
                for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    painter.drawText(int(tx + dx), int(ty + dy), line)
                painter.setPen(zh_color)
                painter.drawText(int(tx), int(ty), line)

        painter.end()

    # --- drag mode ---

    def mouseDoubleClickEvent(self, _ev) -> None:
        self._drag_mode = not self._drag_mode
        self.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents,
            not self._drag_mode,
        )
        if not self._drag_mode:
            self._persist_position()
            self._current_en = ""
            self._current_zh = ""
            self._opacity_value = 0.0
        else:
            self._current_en = "drag mode"
            self._current_zh = "(drop & double-click to lock)"
            self._opacity_value = 1.0
        self.update()

    def mousePressEvent(self, ev) -> None:
        if not self._drag_mode:
            return
        if ev.button() == Qt.MouseButton.LeftButton:
            self._drag_offset = (
                ev.globalPosition().toPoint() - self.frameGeometry().topLeft()
            )

    def mouseMoveEvent(self, ev) -> None:
        if not self._drag_mode or self._drag_offset is None:
            return
        if ev.buttons() & Qt.MouseButton.LeftButton:
            self.move(ev.globalPosition().toPoint() - self._drag_offset)

    def mouseReleaseEvent(self, _ev) -> None:
        self._drag_offset = None

    def _persist_position(self) -> None:
        pos = self.pos()
        self.config.subtitle.position.x = pos.x()
        self.config.subtitle.position.y = pos.y()
        try:
            self.config.save(self.config_path)
            logger.info(f"Saved subtitle position: ({pos.x()}, {pos.y()})")
        except Exception as e:
            logger.warning(f"Failed to persist subtitle position: {e}")

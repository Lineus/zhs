"""答题过程中的快捷键监听

用户按 ``p`` 表示跳过当前作业，继续下一个作业。
仅在 stdin 是终端时启用；管道 / CI 环境下自动禁用。
"""

import sys
import threading
import time
from typing import Any


class SkipInput:
    """后台监听 stdin，按 ``p`` 请求跳过当前作业

    - Windows：``msvcrt`` 直接读取按键，按 p 即生效
    - POSIX：cbreak 模式逐字符读取，按 p 即生效
    - stdin 不是终端（管道、CI、pytest）时不启动线程
    """

    def __init__(self, key: str = "p") -> None:
        self._key = key.lower()
        self._skip = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._enabled = False

    @property
    def enabled(self) -> bool:
        """监听器是否已启用（stdin 是终端）"""
        return self._enabled

    def start(self) -> None:
        """启动监听线程（stdin 非终端时静默跳过）"""
        try:
            if sys.stdin is None or not sys.stdin.isatty():
                return
        except (AttributeError, OSError, ValueError):
            return

        self._enabled = True
        self._thread = threading.Thread(target=self._run, name="zhs-skip-input", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        if sys.platform == "win32":
            self._run_windows()
        else:
            self._run_posix()

    def _run_windows(self) -> None:
        import msvcrt

        while not self._stop.is_set():
            if msvcrt.kbhit():
                char = msvcrt.getwch()
                if char.lower() == self._key:
                    self._skip.set()
                    return
            time.sleep(0.05)

    def _run_posix(self) -> None:
        import select
        import termios
        import tty

        # mypy 在 Windows 下不会识别 termios/tty 的 Unix 属性
        termios_mod: Any = termios
        tty_mod: Any = tty

        fd = sys.stdin.fileno()
        try:
            old_settings = termios_mod.tcgetattr(fd)
        except termios_mod.error:
            return

        try:
            tty_mod.setcbreak(fd)
            while not self._stop.is_set():
                readable, _, _ = select.select([sys.stdin], [], [], 0.1)
                if not readable:
                    continue
                char = sys.stdin.read(1)
                if char.lower() == self._key:
                    self._skip.set()
                    return
        finally:
            termios_mod.tcsetattr(fd, termios_mod.TCSADRAIN, old_settings)

    def consume(self) -> bool:
        """是否请求跳过；返回 True 时清除标记（只触发一次）"""
        if self._skip.is_set():
            self._skip.clear()
            return True
        return False

    def stop(self) -> None:
        """停止监听并恢复终端设置"""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)
            self._thread = None
        self._enabled = False


__all__ = ["SkipInput"]

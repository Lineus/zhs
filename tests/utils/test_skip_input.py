"""utils/skip_input.py 测试"""

import sys
from unittest.mock import MagicMock, patch

from zhs.utils.skip_input import SkipInput


class TestConsume:
    """跳过标记消费"""

    def test_consume_once(self) -> None:
        """标记只能消费一次"""
        skip = SkipInput()
        assert skip.consume() is False

        skip._skip.set()
        assert skip.consume() is True
        assert skip.consume() is False


class TestStart:
    """启动监听"""

    def test_disabled_when_stdin_not_tty(self) -> None:
        """stdin 不是终端时不启动线程"""
        skip = SkipInput()
        fake_stdin = MagicMock()
        fake_stdin.isatty.return_value = False

        with patch("zhs.utils.skip_input.sys.stdin", fake_stdin):
            skip.start()

        assert skip.enabled is False
        assert skip._thread is None

    def test_enabled_when_stdin_is_tty(self) -> None:
        """stdin 是终端时启动后台线程"""
        skip = SkipInput()
        fake_stdin = MagicMock()
        fake_stdin.isatty.return_value = True

        with (
            patch("zhs.utils.skip_input.sys.stdin", fake_stdin),
            patch.object(skip, "_run"),
        ):
            skip.start()
            assert skip.enabled is True
            assert skip._thread is not None
            skip.stop()

        assert skip.enabled is False

    def test_stop_without_start(self) -> None:
        """未启动时 stop 不抛异常"""
        SkipInput().stop()


class TestWindowsReader:
    """Windows 按键读取"""

    def test_p_sets_skip_flag(self) -> None:
        """读取到 p 时设置跳过标记"""
        skip = SkipInput()
        fake_msvcrt = MagicMock()
        fake_msvcrt.kbhit.return_value = True
        fake_msvcrt.getwch.return_value = "P"

        with patch.dict(sys.modules, {"msvcrt": fake_msvcrt}):
            skip._run_windows()

        assert skip.consume() is True

    def test_stop_exits_loop(self) -> None:
        """stop 后循环退出"""
        skip = SkipInput()
        fake_msvcrt = MagicMock()
        fake_msvcrt.kbhit.side_effect = lambda: skip._stop.set() or False

        with patch.dict(sys.modules, {"msvcrt": fake_msvcrt}):
            skip._run_windows()

        assert skip.consume() is False

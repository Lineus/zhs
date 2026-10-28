"""CLI 基础设施：日志、代理、配置加载、Cookie 恢复、登录

从原 __main__.py 抽离的非业务逻辑：
- setup_logger: 配置 loguru
- parse_proxy: 解析代理字符串
- load_config_and_session: 加载配置 + 创建 session + 恢复 cookies
- try_restore_cookies: 从本地文件恢复 cookies
- do_login: 执行扫码登录
"""

import json
import sys
from collections.abc import Callable
from typing import Any

import httpx
import typer
from loguru import logger

from zhs.config import AppConfig, ConfigManager
from zhs.exceptions import ApiError, CaptchaRequired
from zhs.session import ZhsSession


def setup_logger(config: AppConfig, debug: bool, console_log: bool) -> None:
    """配置日志"""
    log_level = "DEBUG" if debug else config.display.log_level
    logger.remove()
    from zhs.utils.path import get_data_dir

    log_dir = get_data_dir() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file_pattern = str(log_dir / "zhs_{time:YYYY-MM-DD}.log")
    logger.add(
        log_file_pattern,
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {thread.name} | {name}:{function}:{line} | {message}",
        rotation="00:00",
        retention="30 days",
        compression="gz",
        encoding="utf-8",
    )
    if console_log or debug:
        console_fmt = "<level>{level:<7}</level> | <cyan>{name}</cyan> - {message}"
        logger.add(sys.stderr, level="INFO", format=console_fmt)
    logger.info(f"日志目录: {log_dir}")


def parse_proxy(config: AppConfig, proxy: str) -> None:
    """解析代理字符串"""
    parts = proxy.lower().split("://")
    if len(parts) != 2:
        logger.error(f"不支持的代理格式: {proxy}")
        return

    schema, _ = parts
    if schema in ("http", "https", "socks5"):
        config.proxies.http = proxy
        config.proxies.https = proxy
    else:
        logger.error(f"不支持的代理类型: {schema}")


def check_ai_provider(config: AppConfig) -> None:
    """启动时提示 AI 智慧课程实际生效的提供方（仅提示，不改变行为）

    `use_builtin_ai`（默认 true）只作用于 AI 智慧课程：为 true 时使用智慧树内置 AI，
    此时已配置的自定义 api_key 不会被 AI 智慧课程使用（但知到作业/考试仍会用它，
    因为它们没有内置 AI 可用）。
    """
    ai = config.ai
    if ai.enabled and ai.use_builtin_ai and ai.api_key:
        logger.warning(
            "检测到已配置自定义 API Key 但 use_builtin_ai=true，AI 智慧课程将使用智慧树内置 AI；"
            "如需 AI 智慧课程也用自定义模型请设 use_builtin_ai=false"
            "（知到作业/考试没有内置 AI，仍会使用该 API Key）"
        )


def load_config_and_session(debug: bool, console_log: bool, proxy: str | None) -> tuple[AppConfig, ZhsSession] | None:
    """加载配置、创建 session、恢复 cookies。失败返回 None。"""
    config_mgr = ConfigManager()
    config = config_mgr.load()

    if proxy:
        parse_proxy(config, proxy)

    setup_logger(config, debug, console_log)
    check_ai_provider(config)

    session = ZhsSession(config)

    reasons: list[str] = []
    if not try_restore_cookies(session, config, on_error=reasons.append):
        from zhs.utils.display import msg_error, msg_info

        reason = reasons[0] if reasons else ""
        if reason:
            print(msg_error(f"登录状态无效：{reason}"))
            if "验证码" in reason:
                print(msg_info("Cookie 可能仍然有效，请先在网页端完成验证码后再运行"))
            else:
                print(msg_info("请运行 zhs login 重新登录"))
        else:
            print(msg_error("未登录或 Cookie 已过期，请先运行 zhs login 登录"))
        return None

    return config, session


def try_restore_cookies(
    session: ZhsSession,
    config: AppConfig,
    on_error: Callable[[str], None] | None = None,
) -> bool:
    """尝试恢复 cookies，成功返回 True

    失败原因通过 ``on_error`` 回调返回，便于 CLI 展示具体原因而不是笼统的
    “未登录或 Cookie 已过期”。

    注意：知到课程列表为空 **不等于** 未登录（账号可能只有 Hike/AI 课程）。
    未登录时课程列表接口会返回 HTTP 401，据此区分。
    """

    def fail(reason: str) -> bool:
        logger.warning(f"Cookie 恢复失败: {reason}")
        if on_error is not None:
            on_error(reason)
        return False

    if not config.save_cookies:
        return fail("配置中 save_cookies = false，登录态不会保存")

    from zhs.utils.cookie import list_to_cookies
    from zhs.utils.path import get_data_dir

    cookies_path = get_data_dir() / "cookies.json"
    if not cookies_path.exists():
        return fail(f"未找到 Cookie 文件 {cookies_path.resolve()}，请先运行 zhs login")

    try:
        with open(cookies_path, encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        return fail(f"Cookie 文件无法读取或已损坏（{cookies_path.resolve()}）：{e}")

    try:
        session.cookies = list_to_cookies(raw)
    except (KeyError, TypeError, ValueError) as e:
        return fail(f"Cookie 文件格式不正确（{cookies_path.resolve()}）：{e}")

    # 验证 cookies 有效性：401 才是未登录，空课程列表可能是账号没有知到课程
    from zhs.zhidao.course import ZhidaoCourseManager

    try:
        courses = ZhidaoCourseManager(session).get_course_list()
    except CaptchaRequired:
        return fail("服务端要求验证码（-12），请在网页端完成验证码后重试")
    except httpx.HTTPStatusError as e:
        status = e.response.status_code
        if status in (401, 403):
            return fail(f"服务端返回 HTTP {status}，Cookie 已失效")
        return fail(f"无法验证登录状态：HTTP {status}")
    except ApiError as e:
        return fail(f"服务端返回错误 {e.code}：{e.message}")
    except Exception as e:
        return fail(f"无法验证登录状态：{type(e).__name__}: {e}")

    if courses:
        logger.info(f"Cookie 恢复成功（{len(courses)} 门知到课程）")
    else:
        logger.info("Cookie 恢复成功（当前账号没有知到共享课程，可能只有 Hike/AI 课程）")
    from zhs.utils.display import msg_done

    print(msg_done("登录状态有效"))
    return True


def verify_login(session: ZhsSession) -> tuple[bool, str]:
    """验证登录态是否真的可用（与启动时的校验保持一致）

    Returns:
        (是否通过, 说明)。通过但需要提示时说明非空。
    """
    from zhs.zhidao.course import ZhidaoCourseManager

    try:
        courses = ZhidaoCourseManager(session).get_course_list()
    except CaptchaRequired:
        return True, "服务端要求验证码（-12），Cookie 已保存；请先在网页端完成验证码"
    except httpx.HTTPStatusError as e:
        status = e.response.status_code
        if status in (401, 403):
            return False, f"服务端返回 HTTP {status}，登录态未建立"
        return True, f"无法验证登录状态：HTTP {status}（Cookie 已保存）"
    except ApiError as e:
        return True, f"无法验证登录状态：服务端错误 {e.code}（Cookie 已保存）"
    except Exception as e:
        return True, f"无法验证登录状态：{type(e).__name__}: {e}（Cookie 已保存）"

    if courses:
        logger.info(f"登录态验证通过（{len(courses)} 门知到课程）")
    else:
        logger.info("登录态验证通过（当前账号没有知到共享课程，可能只有 Hike/AI 课程）")
    return True, ""


def do_login(
    login_mgr: Any,
    config: AppConfig,
    show_in_terminal: bool,
    session: ZhsSession | None = None,
) -> None:
    """执行扫码登录

    Args:
        login_mgr: LoginManager 实例
        config: 应用配置
        show_in_terminal: 是否在终端显示二维码
        session: 传入时在登录后验证登录态，避免“登录成功但下一次命令提示未登录”
    """
    from zhs.utils.display import msg_error, msg_info, msg_warn
    from zhs.utils.display import show_qrcode_img as _show_qr_img

    def qr_callback(img_bytes: bytes) -> None:
        if show_in_terminal:
            _show_qr_img(img_bytes)

    result = login_mgr.login_with_qr(qr_callback, image_path=config.qr.image_path)
    if not result.success or not result.cookies:
        logger.error("登录失败：未获取到有效 cookies")
        raise typer.Exit(1)

    logger.info("登录成功")

    # 保存 cookies
    if config.save_cookies:
        from zhs.utils.cookie import cookies_to_list
        from zhs.utils.path import get_data_dir

        cookies_path = get_data_dir() / "cookies.json"
        with open(cookies_path, "w", encoding="utf-8") as f:
            json.dump(cookies_to_list(result.cookies), f, indent=2, ensure_ascii=False)
        logger.info(f"Cookie 已保存到 {cookies_path}")
    else:
        logger.warning("save_cookies = false，本次登录态未保存，下次运行需要重新登录")
        print(msg_warn("save_cookies = false，登录态未保存，下次运行需要重新登录"))

    if session is not None:
        ok, note = verify_login(session)
        if not ok:
            print(msg_error(f"登录流程已完成，但登录态验证失败：{note}"))
            print(msg_info("请重新运行 zhs login；若仍失败，请检查账号状态或网络"))
            raise typer.Exit(1)
        if note:
            print(msg_warn(note))


def init_llm(config: AppConfig) -> Any:
    """初始化 LLM 提供者（知到作业/知到考试专用）

    智慧树内置 AI 仅用于 AI 智慧课程（由 LLMProviderFactory 创建 ZhidaoAIProvider），
    知到作业/考试没有内置 AI 可用，只能用自定义 API Key（与 use_builtin_ai 无关）。
    此处仅创建 OpenAI 兼容 provider；无 api_key 则返回 None。

    Returns:
        LLMProvider | None
    """
    from zhs.llm.openai import OpenAIProvider

    ai = config.ai
    if not ai.enabled:
        return None
    if not ai.api_key:
        logger.warning("API key 为空，LLM 不可用，将使用随机答题")
        return None
    return OpenAIProvider(
        api_key=ai.api_key,
        base_url=ai.base_url,
        model_name=ai.model,
        max_token=ai.max_token,
        extra_headers=ai.extra_headers,
    )


def init_question_bank(config: AppConfig, scope: str, llm: Any = None) -> Any:
    """初始化题库查询客户端

    题库依赖 LLM：题库答案仅作为 LLM 答题的提示源，无 LLM 时题库无用。
    若 LLM 未初始化（llm is None）则禁用题库并告警。
    scope 决定当前场景是否在启用范围内（zhidao_homework/zhidao_exam/ai_homework/ai_exam）。

    Args:
        config: 应用配置
        scope: 题库启用范围
        llm: 已初始化的 LLM 实例（None=未初始化）

    Returns:
        QuestionBankClient | None
    """
    from zhs.question_bank.client import QuestionBankClient
    from zhs.utils.display import msg_warn

    qb = config.question_bank
    if not qb.enabled:
        return None
    if not qb.token:
        logger.warning("题库 token 为空，题库不可用")
        print(msg_warn("题库已启用但 token 为空，题库不可用（请在配置中设置 token 或关闭题库功能）"))
        return None
    if llm is None:
        logger.warning("题库功能依赖 LLM，但 LLM 未初始化（请检查 ai.api_key 配置），已禁用题库查询")
        print(msg_warn("题库功能依赖 LLM，但 LLM 未初始化，已禁用题库查询"))
        return None
    if scope not in qb.scopes:
        return None
    return QuestionBankClient(
        token=qb.token,
        query_url=qb.query_url,
        info_url=qb.info_url,
    )


__all__ = [
    "do_login",
    "init_llm",
    "init_question_bank",
    "load_config_and_session",
    "parse_proxy",
    "setup_logger",
    "try_restore_cookies",
    "verify_login",
]

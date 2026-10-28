"""AI 作业/考试基类（模板方法模式）

封装 HomeworkCtx 与 ExamCtx 的公共流程与状态：
- LLM 提供者初始化（通过 LLMProviderFactory）
- 单级内存缓存（_answer_cache，当前 exam），持久化通过 AiExamCache
- 三级答案策略（缓存 → AI → 随机）
- 心跳（daemon 线程）
- 进度条更新
- 结果检查与缓存更新

缓存查询顺序（_get_cached_answer）：
1. _answer_cache（当前 exam 内存缓存）
2. cache.get（当前 exam SQLite 查询）
3. cache.search_in_course（跨 exam 课程级查询，仅 ExamCtx 启用）

子类只需实现差异化的抽象方法（_open / _get_sheet_content / _save_answer 等），
并通过 _cross_exam_search 属性控制是否启用跨 exam 查询。
"""

import contextlib
import random
import re
import threading
import time
from abc import ABC, abstractmethod
from typing import Any

from loguru import logger

from zhs.ai.models import OptionVo, QuestionContent, QuestionSheet
from zhs.cache.ai_cache import AiExamCache
from zhs.config import AIConfig
from zhs.exceptions import ZhsError
from zhs.llm.base import LLMProvider
from zhs.llm.factory import LLMProviderFactory
from zhs.question_bank.client import QuestionBankClient
from zhs.question_bank.models import map_question_type
from zhs.reporter import ConsoleReporter, ProgressReporter
from zhs.session import ZhsSession
from zhs.utils.display import _C, progress_bar, styled
from zhs.utils.html import extract_text


class AiExamBase(ABC):
    """AI 作业/考试基类（模板方法模式）

    子类 HomeworkCtx（逐题保存）与 ExamCtx（批量保存）共享此基类。
    """

    def __init__(
        self,
        session: ZhsSession,
        course_id: int | str,
        exam_test_id: int | str,
        exam_paper_id: int | str,
        ai_config: AIConfig,
        op_extra: dict[str, Any] | None = None,
        progress_view: bool = True,
        reporter: ProgressReporter | None = None,
        cache: AiExamCache | None = None,
        question_bank: QuestionBankClient | None = None,
        reference_answers: dict[int, str] | None = None,
    ) -> None:
        self._session = session
        self._course_id = course_id
        self._exam_test_id = exam_test_id
        self._exam_paper_id = exam_paper_id
        self._ai_config = ai_config
        self._op_extra = op_extra or {}
        self._progress_view = progress_view
        self._reporter = reporter or ConsoleReporter()
        self._cache = cache or AiExamCache()
        self._question_bank = question_bank
        # 平台错题接口返回的标准答案（question_id -> result），优先级高于缓存
        self._reference_answers: dict[int, str] = reference_answers or {}

        # 缓存（当前 exam 内存缓存，持久化通过 self._cache）
        self._answer_cache: dict[str, dict[str, Any]] = {}
        # 本次答题中来自缓存的题目 ID（用于提交后判断缓存答案是否被平台拒绝）
        self._cached_answer_ids: set[int] = set()
        # 本次实际提交的答案字符串（用于答对后原样写回缓存）
        self._submitted_answer_str: dict[int, str] = {}

        # 状态
        self._sheet_content: list[QuestionSheet] | None = None
        self._stopped = False
        self._heartbeat_thread: threading.Thread | None = None
        self._reference_materials: list[dict[str, str]] = []

        # 进度
        self._progress_total = 0
        self._progress_current = 0
        self._progress_sources: list[str] = []

        # LLM 提供者（统一通过工厂创建）
        self._provider: LLMProvider | None = LLMProviderFactory.create(
            ai_config,
            session=session,
            course_id=str(course_id),
            course_name=str(self._op_extra.get("courseName", "")) if self._op_extra else "",
        )

    # --- 模板方法 ---

    def start(
        self,
        reference_materials: list[dict[str, str]] | None = None,
        submit: bool = False,
    ) -> tuple[bool, int, int]:
        """执行完整流程，返回 (是否全对, 正确数, 总题数)

        Args:
            reference_materials: 参考资料（PPT 等）
            submit: 是否提交（子类决定是否使用）
        """
        self._reference_materials = reference_materials or []
        self._cached_answer_ids = set()

        self._load_cache()
        self._open()

        # 启动心跳（daemon 线程）
        self._heartbeat_thread = threading.Thread(target=self._heartbeat, daemon=True)
        self._heartbeat_thread.start()

        sheets = self._get_sheet_content()
        if not sheets:
            self._stopped = True
            raise ZhsError("答题卡内容为空")

        # 答题
        total = len(sheets)
        self._progress_total = total
        self._progress_current = 0
        self._progress_sources = []
        self._update_progress("pending")
        self._answer_questions(sheets)

        self._reporter.wipe_line()

        # 提交与结果检查（子类决定具体行为）
        result = self._finish(submit, sheets)
        self._stopped = True
        return result

    def _finish(self, submit: bool, sheets: list[QuestionSheet]) -> tuple[bool, int, int]:
        """结束流程（提交 + 结果检查）。默认实现：不提交，返回 (False, 0, total)。

        子类可重写以实现提交逻辑。
        """
        return False, 0, len(sheets)

    # --- 公共实现（子类一般不重写）---

    @property
    @abstractmethod
    def _exam_base_url(self) -> str:
        """考试 API 基础 URL"""
        ...

    @abstractmethod
    def _api_query(self, url: str, data: dict[str, Any], method: str = "POST") -> dict[str, Any]:
        """同步 API 查询（子类决定使用哪个 session 方法）"""
        ...

    @abstractmethod
    def _open(self) -> None:
        """打开作业/考试"""
        ...

    @abstractmethod
    def _get_sheet_content(self) -> list[QuestionSheet]:
        """获取答题卡内容"""
        ...

    @abstractmethod
    def _get_question_content(self, question_id: int, version: int) -> QuestionContent | None:
        """获取题目详情"""
        ...

    @abstractmethod
    def _save_answer(self, question_id: int, answers: list[str]) -> bool:
        """保存单题答案"""
        ...

    @abstractmethod
    def _submit(self, submit: bool) -> None:
        """提交"""
        ...

    @abstractmethod
    def _answer_questions(self, sheets: list[QuestionSheet]) -> None:
        """答题流程（子类决定逐题或批量）"""
        ...

    def _lookup_reference_answer(self, question: QuestionContent) -> str | None:
        """查找平台标准答案（默认只查构造函数传入的映射）

        HomeworkCtx 重写此方法，按需调用平台接口获取单题标准答案。
        """
        return self._reference_answers.get(question.id)

    def _lookup_reference_option_answer(self, question: QuestionContent) -> list[str] | None:
        """查找平台标准选项（选择题/判断题）

        默认不启用；HomeworkCtx 重写此方法，从题库获取正确选项内容并按内容
        映射到当前试卷的选项 ID。
        """
        return None

    # --- 缓存键 ---

    @staticmethod
    def _cache_key(question_id: int) -> str:
        """生成缓存键（仅用 question_id，不含 version）"""
        return str(question_id)

    # --- 跨 exam 查询开关 ---

    @property
    def _cross_exam_search(self) -> bool:
        """是否启用跨 exam 课程级查询（ExamCtx 重写为 True，HomeworkCtx 默认 False）

        - homework 仅查当前 exam（作业针对单个考试，题目已知）
        - exam 可跨 exam 搜索课程题库（考试考察课程全部知识，需复用其他 exam 的答案）
        """
        return False

    # --- 缓存查询 ---

    @staticmethod
    def _option_content_key(content: str) -> str:
        """生成选项的稳定指纹（忽略 HTML 包装/空白，保留图片 src）

        每次重做作业时平台会重新生成 paper，选项 ID 会整体偏移，但选项内容
        （文本或图片 URL/base64）保持不变，因此可以用内容把旧 ID 映射到新 ID。
        """
        if not content:
            return ""
        images = re.findall(r'<img[^>]+src="([^"]+)"', content)
        text = extract_text(content).strip()
        if images:
            return "img:" + "|".join(images) + "|text:" + text
        return "text:" + text

    def _map_cached_answer_to_options(
        self, entry: dict[str, Any], question: QuestionContent | None
    ) -> list[str] | None:
        """把缓存里的旧选项 ID 映射到当前试卷的选项 ID

        Returns:
            - 列表：映射成功（可能为空，表示映射失败、应视为未命中）
            - None：缓存缺少旧选项信息，交给原始 ID 兜底
        """
        if question is None or question.question_type not in (1, 2, 14) or not question.option_vos:
            return None

        cached_ids = AiExamCache.parse_answer(entry.get("answer", ""))
        if not cached_ids:
            return None

        question_dict = entry.get("questionDict")
        old_options = question_dict.get("option_vos") if isinstance(question_dict, dict) else None
        if not isinstance(old_options, list):
            return None

        old_by_id = {str(o.get("id")): o for o in old_options if isinstance(o, dict)}
        current_by_key: dict[str, list[str]] = {}
        for opt in question.option_vos:
            key = self._option_content_key(opt.content)
            current_by_key.setdefault(key, []).append(str(opt.id))

        mapped: list[str] = []
        for cached_id in cached_ids:
            old = old_by_id.get(str(cached_id))
            if old is None:
                # 旧选项数据不完整，缓存不可信
                return []
            key = self._option_content_key(str(old.get("content") or ""))
            candidates = current_by_key.get(key, [])
            if len(candidates) != 1:
                # 内容未找到或不唯一，无法安全映射
                return []
            mapped.append(candidates[0])
        return mapped

    def _resolve_cached_entry(self, entry: dict[str, Any], question: QuestionContent | None) -> list[str] | None:
        """解析缓存条目，并优先按选项内容映射到当前试卷的选项 ID"""
        mapped = self._map_cached_answer_to_options(entry, question)
        if mapped is not None:
            return mapped or None
        return self._parse_cached_answer(entry.get("answer", ""))

    def _get_cached_answer(
        self,
        question_id: int,
        question_text: str = "",
        question: QuestionContent | None = None,
    ) -> list[str] | None:
        """从缓存获取答案（三级查询）

        查询顺序：
        1. _answer_cache（当前 exam 内存缓存）
        2. cache.get（当前 exam SQLite 查询，内存未命中时的安全兜底）
        3. cache.search_in_course（跨 exam 课程级查询，仅当 question_text 非空）

        Args:
            question_id: 题目 ID
            question_text: 题目文本（清洗后），用于跨 exam 搜索。为空则不跨 exam 搜索。
            question: 当前题目（传入后会按选项内容把旧选项 ID 映射到当前试卷）
        """
        key = self._cache_key(question_id)

        # 1. 当前 exam 内存缓存（空答案视为未命中，继续向后查找）
        entry = self._answer_cache.get(key)
        if entry is not None:
            parsed = self._resolve_cached_entry(entry, question)
            if parsed is not None:
                return parsed

        # 2. 当前 exam SQLite 查询（内存未命中时的安全兜底，回填内存缓存）
        entry = self._cache.get(self._course_id, self._exam_test_id, question_id)
        if entry is not None:
            parsed = self._resolve_cached_entry(entry, question)
            if parsed is not None:
                self._answer_cache[key] = entry
                return parsed

        # 3. 跨 exam 课程级查询（仅 exam 模块，question_text 非空时触发）
        if question_text:
            entry = self._cache.search_in_course(self._course_id, question_text)
            if entry is not None:
                parsed = self._resolve_cached_entry(entry, question)
                if parsed is not None:
                    self._answer_cache[key] = entry
                    return parsed

        return None

    @staticmethod
    def _parse_cached_answer(answer_str: str) -> list[str] | None:
        """解析缓存中的 answer 字段（委托给 AiExamCache.parse_answer）

        - 含 #@# → 按 #@# 分隔（多选题选项 ID）
        - 不含 #@# → 返回单元素列表（单选/判断/填空题）
        填空题多个空用 / 合并存储，不拆分。
        """
        return AiExamCache.parse_answer(answer_str)

    def _set_cached_answer(self, question_id: int, data: dict[str, Any]) -> None:
        """设置缓存（写入当前 exam 内存缓存，_save_cache 时持久化到 SQLite）"""
        key = self._cache_key(question_id)
        cache_entry = {
            "question": data.get("question", ""),
            "answer": data.get("answer", ""),
            "answer_content": data.get("answer_content", ""),
            "questionDict": data.get("questionDict", {}),
        }
        self._answer_cache[key] = cache_entry

    # --- 三级答案策略 ---

    def _get_answer(self, question: QuestionContent) -> tuple[list[str], str]:
        """获取答案，返回 (答案列表, 来源标记)

        三级策略：1. 缓存命中 2. AI 生成 3. 兜底随机
        """
        question_id = question.id
        question_type = question.question_type

        # 0a. 平台标准选项（选择题/判断题，按内容映射到当前试卷选项 ID）
        if question_type in (1, 2, 14):
            platform_options = self._lookup_reference_option_answer(question)
            if platform_options:
                logger.info(f"使用平台标准选项答案: question_id={question_id}")
                return platform_options, "reference"

        # 0b. 平台标准答案（填空题，含公式/图片的标准写法）
        if question_type in (3, 15):
            reference = self._lookup_reference_answer(question)
            if reference:
                logger.info(f"使用平台标准答案: question_id={question_id}")
                return [reference], "reference"

        # 1. 查缓存（exam 模块启用跨 exam 搜索，传清洗后的题目文本）
        question_text = extract_text(question.content) if self._cross_exam_search else ""
        cached = self._get_cached_answer(question_id, question_text, question)
        if cached is not None:
            self._cached_answer_ids.add(question_id)
            return cached, "cached"

        # 选项
        choices = [{"id": opt.id, "content": opt.content} for opt in question.option_vos]

        # 选项少于 2 个且非填空 → 选第一个
        if len(choices) < 2 and question_type != 3:
            return [str(choices[0]["id"])], "cached"

        # 2. AI 生成
        if self._provider is not None:
            extra = self._build_extra(question, choices)
            try:
                if question_type == 1:
                    ids = self._provider.single_choice(question.content, choices, self._reference_materials, extra)
                    if ids:
                        return [str(i) for i in ids], "AI generated"
                    logger.warning(f"{question.content} {question_id} AI provide empty")
                elif question_type == 2:
                    ids = self._provider.multiple_choice(question.content, choices, self._reference_materials, extra)
                    if ids:
                        return [str(i) for i in ids], "AI generated"
                elif question_type == 14:
                    ids = self._provider.judgement(question.content, choices, self._reference_materials, extra)
                    if ids:
                        return [str(i) for i in ids], "AI generated"
                elif question_type == 3:
                    answers = self._provider.fill_blank(question.content, self._reference_materials, extra)
                    if answers:
                        return answers, "AI generated"
            except Exception as e:
                logger.error(f"AI 生成答案失败: {e}")

        # 3. 兜底随机
        if question_type == 3:
            return ["未知"], "random"
        elif question_type == 1 and choices:
            return [str(random.choice(choices)["id"])], "random"
        elif question_type == 2 and choices:
            n = min(2, len(choices))
            selected = random.sample(choices, n)
            return [str(c["id"]) for c in selected], "random"
        elif question_type == 14 and choices:
            return [str(random.choice(choices)["id"])], "random"
        return [], "random"

    # --- 题库参考注入 ---

    def _build_extra(self, question: QuestionContent, choices: list[dict[str, Any]]) -> dict[str, Any]:
        """构建 extra 字典：拷贝 _op_extra，并在题库命中时注入 题库参考。

        题库依赖 AI：仅当 _provider 非空时才查询（无 AI 时查了也无用）。
        题库查询失败/无答案均不影响主流程，返回不含 题库参考 的 extra。
        """
        extra = dict(self._op_extra)
        if self._question_bank is not None:
            hint = self._query_question_bank(question, choices)
            if hint:
                extra["题库参考"] = hint
        return extra

    def _query_question_bank(self, question: QuestionContent, choices: list[dict[str, Any]]) -> str:
        """查询题库，返回格式化提示文本（失败/无答案返回空串）。

        绝不中断做题：所有异常均捕获后返回空串。
        """
        assert self._question_bank is not None
        try:
            title = extract_text(question.content)
            options = "\n".join(extract_text(c["content"]) for c in choices)
            qtype = map_question_type(question.question_type)
            result = self._question_bank.query(title, options, qtype)
            if result is None:
                return ""
            return result.format_hint()
        except Exception as e:
            logger.error(f"题库查询失败: {e}")
            return ""

    # --- 进度条 ---

    def _update_progress(self, source: str) -> None:
        """更新控制台底部进度条"""
        current = self._progress_current
        total = self._progress_total

        # 来源统计
        cache_count = self._progress_sources.count("cached")
        reference_count = self._progress_sources.count("reference")
        ai_count = sum(1 for s in self._progress_sources if s == "AI generated")
        random_count = sum(1 for s in self._progress_sources if s == "random")

        # 来源标签
        source_parts: list[str] = []
        if reference_count:
            source_parts.append(styled(f"ref:{reference_count}", _C.GREEN))
        if cache_count:
            source_parts.append(styled(f"cache:{cache_count}", _C.CYAN))
        if ai_count:
            source_parts.append(styled(f"AI:{ai_count}", _C.BRIGHT_MAGENTA))
        if random_count:
            source_parts.append(styled(f"random:{random_count}", _C.YELLOW))
        source_str = " ".join(source_parts) if source_parts else ""

        # 当前题来源
        if source == "reference":
            cur_tag = styled("[ref]", _C.GREEN)
        elif source == "cached":
            cur_tag = styled("[cache]", _C.CYAN)
        elif source == "AI generated":
            cur_tag = styled("[AI]", _C.BRIGHT_MAGENTA)
        elif source == "random":
            cur_tag = styled("[random]", _C.YELLOW)
        elif source == "error":
            cur_tag = styled("[error]", _C.RED)
        else:
            cur_tag = styled(f"{total}题", _C.DIM)

        bar = progress_bar(current, total)
        line = f"  {bar} {cur_tag} {source_str}"
        with contextlib.suppress(Exception):
            self._reporter.progress(line)

    # --- 心跳 ---

    def _heartbeat(self, interval: int = 10) -> None:
        """同步心跳（daemon 线程），定期更新用时

        默认实现：循环调用 _heartbeat_once，间隔 interval 秒。
        子类可重写 _heartbeat_once 实现具体心跳逻辑。
        """
        while not self._stopped:
            try:
                self._heartbeat_once(interval)
            except Exception as e:
                logger.error(f"心跳失败: {e}")
            time.sleep(interval)

    def _heartbeat_once(self, interval: int = 10) -> None:
        """单次心跳（默认实现：调用 updateUserUsedTime）

        子类可重写以自定义心跳逻辑。
        """
        url = f"{self._exam_base_url}/gateway/t/v1/exam/user/updateUserUsedTime"
        data = {
            "examTestId": self._exam_test_id,
            "examPaperId": self._exam_paper_id,
            "heartbeatTime": interval,
        }
        self._api_query(url, data)

    # --- 结果检查 ---

    def _check_results(self, sheets: list[QuestionSheet]) -> tuple[int, int]:
        """检查结果并更新缓存，返回 (正确数, 总题数)"""
        correct_count = 0
        total_count = len(sheets)

        for sheet in sheets:
            question = self._get_question_content(sheet.question_id, sheet.version)
            if question is None:
                continue

            # 检查用户是否答对
            user_answers = question.user_answer_vos
            is_correct = bool(user_answers) and user_answers[0].is_correct == 1
            if is_correct:
                correct_count += 1

            cached_answer = self._answer_cache.get(self._cache_key(sheet.question_id), {}).get("answer", "")

            # 获取正确选项（用于缓存更新）
            correct_opts = [opt for opt in question.option_vos if opt.is_correct == 1]

            # 更新缓存（question / answer_content 用 extract_text 清洗 HTML）
            if question.question_type == 3:
                # 填空题：多个空用 / 合并存储；含图片/公式的参考答案保留原始 HTML
                if correct_opts:
                    answer_str, answer_content_str = self._build_fill_blank_answer(correct_opts)
                elif user_answers:
                    cleaned = [extract_text(a.answer) for a in user_answers if a.is_correct == 1]
                    answer_str = "/".join(cleaned)
                    answer_content_str = "\n".join(cleaned)
                else:
                    answer_str = ""
                    answer_content_str = ""
            else:
                answer_str = "#@#".join(str(opt.id) for opt in correct_opts)
                answer_content_str = "\n".join(extract_text(opt.content) for opt in correct_opts)

            # 答对时优先把“实际提交并通过判定的答案”写回缓存。参考选项的 HTML
            # 渲染结果与判分使用的标准答案可能不同（例如填空答案含 LaTeX），
            # 若用渲染 HTML 覆盖，下一次反而会答错。
            if is_correct:
                submitted = self._submitted_answer_str.get(sheet.question_id)
                if submitted:
                    answer_str = submitted
                    if not answer_content_str:
                        answer_content_str = extract_text(submitted)

            # 缓存答案再次提交后仍被判错：说明平台不接受这个答案（例如填空题参考
            # 答案含公式图片，extract_text 只剩纯文本）。清空答案，让下一轮改走 AI，
            # 避免每次重试都复读同一个被拒绝的答案。
            if (
                not is_correct
                and user_answers
                and sheet.question_id in self._cached_answer_ids
                and cached_answer
                and answer_str == cached_answer
            ):
                logger.warning(f"缓存答案被判错，已失效: question_id={sheet.question_id} answer={answer_str!r}")
                answer_str = ""
                answer_content_str = ""

            self._set_cached_answer(
                sheet.question_id,
                {
                    "question": extract_text(question.content),
                    "answer": answer_str,
                    "answer_content": answer_content_str,
                    "questionDict": question.model_dump(),
                },
            )

        self._save_cache()
        return correct_count, total_count

    @staticmethod
    def _build_fill_blank_answer(correct_opts: list[OptionVo]) -> tuple[str, str]:
        """构造填空题答案，返回 (提交用答案, 展示用文本)

        客观填空题的答题框只能输入纯文本，但参考答案可能是带公式图片的 HTML。
        此时直接提交服务端下发的原始参考答案（其中已包含图片 URL），绕过输入框
        无法插入图片的限制；图片本身无需重新上传。

        多个空用 #@# 分隔，与官方客观填空题的提交格式一致。
        """
        raw_contents = [opt.content for opt in correct_opts]
        cleaned = [extract_text(content) for content in raw_contents]
        display = "\n".join(cleaned)

        if any("<img" in content.lower() for content in raw_contents):
            logger.info("填空题参考答案含图片/公式，使用原始 HTML 作为答案")
            return "#@#".join(raw_contents), display

        return "/".join(cleaned), display

    # --- 缓存持久化（通过 AiExamCache）---

    def _load_cache(self) -> None:
        """加载当前 exam 的缓存到内存（通过 AiExamCache 读取 SQLite）

        仅加载当前 exam，跨 exam 查询通过 search_in_course 按需触发，
        不再预加载课程下所有 exam（去掉 _all_answer_cache 内存合并）。
        """
        self._answer_cache = {}
        exam_entries = self._cache.load_exam(self._course_id, self._exam_test_id)
        for key, value in exam_entries.items():
            if isinstance(value, dict):
                self._answer_cache[key] = value

    def _save_cache(self) -> None:
        """保存当前 exam 内存缓存到 SQLite（通过 AiExamCache.put）

        course_name 从 _op_extra["courseName"] 取，传入 cache.put 以持久化。
        UPSERT 的 CASE WHEN 逻辑会在已有空值时回填，不会覆盖已有非空值。
        """
        course_name = str(self._op_extra.get("courseName", "")) if self._op_extra else ""
        logger.debug(
            f"保存缓存: course_id={self._course_id}, exam_id={self._exam_test_id}, "
            f"course_name={course_name!r}, 条目数={len(self._answer_cache)}"
        )
        for question_id_str, entry in self._answer_cache.items():
            try:
                question_id = int(question_id_str)
                self._cache.put(
                    self._course_id,
                    self._exam_test_id,
                    question_id,
                    entry,
                    course_name=course_name,
                )
            except (ValueError, TypeError) as e:
                logger.error(f"保存缓存失败 question_id={question_id_str}: {e}")

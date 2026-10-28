"""AI 作业上下文（同步执行）

继承 AiExamBase，实现逐题保存的作业流程。
AI 课程的"考试"实际上是作业，统一命名为 homework。

与 ExamCtx 的区别：
- 逐题保存（saveAnswer）而非批量保存
- openExam 包含 examPaperId 字段
- getExamSheetInfo 取 partSheetVos[0]
- submit 包含 courseType=8 和 aiKnlowledgeId
"""

import random
import time
from collections.abc import Callable
from typing import Any

from loguru import logger

from zhs.ai.exam_base import AiExamBase
from zhs.ai.models import QuestionContent, QuestionSheet
from zhs.config import AIConfig
from zhs.exceptions import ZhsError
from zhs.question_bank.client import QuestionBankClient
from zhs.session import ZhsSession
from zhs.utils.skip_input import SkipInput


class HomeworkCtx(AiExamBase):
    """AI 作业上下文（逐题保存）"""

    def __init__(
        self,
        session: ZhsSession,
        course_id: int,
        knowledge_id: int,
        exam_test_id: int,
        exam_paper_id: int,
        ai_config: AIConfig,
        op_extra: dict[str, Any] | None = None,
        progress_view: bool = True,
        question_bank: QuestionBankClient | None = None,
        skip_input: SkipInput | None = None,
        reference_answers: dict[int, str] | None = None,
        reference_fetcher: Callable[[int], str | None] | None = None,
        reference_options_fetcher: Callable[[int], list[str]] | None = None,
    ) -> None:
        super().__init__(
            session=session,
            course_id=course_id,
            exam_test_id=exam_test_id,
            exam_paper_id=exam_paper_id,
            ai_config=ai_config,
            op_extra=op_extra,
            progress_view=progress_view,
            question_bank=question_bank,
            reference_answers=reference_answers,
        )
        self._knowledge_id = knowledge_id
        self._skip_input = skip_input
        self._reference_fetcher = reference_fetcher
        self._reference_options_fetcher = reference_options_fetcher
        self._reference_option_answers: dict[int, list[str]] = {}
        # 用户按 p 跳过当前作业（不再提交、也不再重试）
        self._skipped = False

    @property
    def skipped(self) -> bool:
        """本次作业是否被用户跳过"""
        return self._skipped

    @property
    def used_reference(self) -> bool:
        """本次作业是否有题目使用了平台标准答案"""
        return "reference" in self._progress_sources

    def _lookup_reference_answer(self, question: QuestionContent) -> str | None:
        """优先使用平台下发的标准答案，其次按需调用平台接口查询

        平台 ``get-test-question`` 接口的 ``result`` 字段是判分使用的标准答案；
        对含公式/图片的填空题，它可能是 ``相对运动速度 $$\\vec{v}_r$$`` 这样的
        文本 + LaTeX，而不是渲染后的 HTML。
        """
        reference = super()._lookup_reference_answer(question)
        if reference or self._reference_fetcher is None:
            return reference

        try:
            reference = self._reference_fetcher(question.id)
        except Exception as e:
            logger.warning(f"查询题目标准答案失败: question_id={question.id}: {e}")
            return None

        if reference:
            self._reference_answers[question.id] = reference
        return reference

    def _lookup_reference_option_answer(self, question: QuestionContent) -> list[str] | None:
        """选择题/判断题：查询题库正确选项内容，并映射到当前试卷的选项 ID

        平台每次重做会生成新 paper，选项 ID 会变化；题库返回的选项内容稳定，
        按内容映射可以避免提交到错误选项。
        """
        if question.question_type not in (1, 2, 14) or self._reference_options_fetcher is None:
            return None

        cached = self._reference_option_answers.get(question.id)
        if cached is not None:
            return cached or None

        try:
            contents = self._reference_options_fetcher(question.id)
        except Exception as e:
            logger.warning(f"查询题目标准选项失败: question_id={question.id}: {e}")
            contents = []

        if not contents:
            self._reference_option_answers[question.id] = []
            return None

        current_by_key: dict[str, list[str]] = {}
        for opt in question.option_vos:
            key = self._option_content_key(opt.content)
            current_by_key.setdefault(key, []).append(str(opt.id))

        mapped: list[str] = []
        for content in contents:
            candidates = current_by_key.get(self._option_content_key(content), [])
            if len(candidates) != 1:
                logger.warning(f"题库选项无法映射到当前试卷: question_id={question.id}")
                self._reference_option_answers[question.id] = []
                return None
            mapped.append(candidates[0])

        self._reference_option_answers[question.id] = mapped
        return mapped or None

    def _skip_requested(self) -> bool:
        """检查并消费用户跳过请求"""
        if self._skip_input is not None and self._skip_input.consume():
            logger.info("收到跳过指令，停止作答当前作业")
            self._skipped = True
            return True
        return False

    # --- API ---

    @property
    def _exam_base_url(self) -> str:
        """作业 API 基础 URL"""
        return self._session.urls.exam

    def _api_query(self, url: str, data: dict[str, Any], method: str = "POST") -> dict[str, Any]:
        """同步作业 API 查询"""
        return self._session.ai_exam_query(url, data, method=method)

    def _open(self, _retries: int = 0) -> None:
        """打开作业"""
        url = f"{self._exam_base_url}/gateway/t/v1/exam/user/openExam"
        data = {
            "examTestId": self._exam_test_id,
            "examPaperId": self._exam_paper_id,
            "courseId": self._course_id,
        }
        try:
            self._api_query(url, data)
        except Exception as e:
            if _retries < 2:
                logger.error(f"openExam 失败，重试 {_retries + 1}/3")
                self._open(_retries + 1)
            else:
                raise ZhsError("openExam 失败，已重试 3 次") from e

    def _get_sheet_content(self, _retries: int = 0) -> list[QuestionSheet]:
        """获取试卷内容（getExamSheetInfo，取 partSheetVos[0]）"""
        if self._sheet_content is not None:
            return self._sheet_content

        url = f"{self._exam_base_url}/gateway/t/v1/exam/user/getExamSheetInfo"
        data = {
            "examTestId": self._exam_test_id,
            "examPaperId": self._exam_paper_id,
        }
        try:
            result = self._api_query(url, data, method="GET")
            raw_sheets = result["data"]["partSheetVos"][0]["questionSheetVos"]
            self._sheet_content = [QuestionSheet.model_validate(s) for s in raw_sheets]
        except Exception as e:
            if _retries < 2:
                logger.error(f"getSheetContent 失败，重试 {_retries + 1}/3")
                return self._get_sheet_content(_retries + 1)
            else:
                raise ZhsError("getSheetContent 失败，已重试 3 次") from e

        return self._sheet_content

    def _get_question_content(self, question_id: int, version: int, _retries: int = 0) -> QuestionContent | None:
        """获取题目内容"""
        url = f"{self._exam_base_url}/gateway/t/v1/question/getExamQuestionInfo"
        data = {
            "examTestId": self._exam_test_id,
            "examPaperId": self._exam_paper_id,
            "questionId": question_id,
            "version": version,
        }
        try:
            result = self._api_query(url, data, method="GET")
            return QuestionContent.model_validate(result["data"])
        except Exception as e:
            if _retries < 2:
                logger.error(f"getQuestionContent 失败，重试 {_retries + 1}/3")
                return self._get_question_content(question_id, version, _retries + 1)
            else:
                logger.error(f"getQuestionContent 失败，已重试 3 次: {e}")
                return None

    def _save_answer(self, question_id: int, answers: list[str]) -> bool:
        """保存单题答案（saveAnswer）"""
        if not answers:
            return False
        url = f"{self._exam_base_url}/gateway/t/v1/answer/saveAnswer"
        data = {
            "recruitId": self._course_id,
            "examTestId": self._exam_test_id,
            "examPaperId": self._exam_paper_id,
            "questionId": question_id,
            "dataVos": None,
            "answer": "#@#".join(str(a) for a in answers),
        }
        try:
            self._api_query(url, data)
            return True
        except Exception as e:
            logger.error(f"saveAnswer 失败: {e}")
            return False

    def _submit(self, submit: bool = True, _retries: int = 0) -> None:
        """提交作业（含 courseType=8 和 aiKnlowledgeId）"""
        url = f"{self._exam_base_url}/gateway/t/v1/exam/user/submit"
        data = {
            "examTestId": self._exam_test_id,
            "courseId": self._course_id,
            "courseType": 8,
            "examPaperId": self._exam_paper_id,
            "aiKnlowledgeId": self._knowledge_id,
        }
        try:
            self._api_query(url, data)
        except Exception as e:
            if _retries < 2:
                logger.error(f"submitExam 失败，重试 {_retries + 1}/3")
                self._submit(submit, _retries + 1)
            else:
                raise ZhsError("submitExam 失败，已重试 3 次") from e
        finally:
            self._stopped = True

    def _answer_questions(self, sheets: list[QuestionSheet]) -> None:
        """逐题顺序处理（含延迟防限流）"""
        for sheet in sheets:
            if self._skipped or self._skip_requested():
                return
            self._process_question(sheet)

    def _process_question(self, sheet: QuestionSheet) -> None:
        """处理单道题目（顺序执行，含延迟防限流）"""
        time.sleep(random.uniform(3, 5))

        # 延迟期间用户可能按了 p，避免再发起一次 LLM 请求
        if self._skipped or self._skip_requested():
            return

        question = self._get_question_content(sheet.question_id, sheet.version)
        if question is None:
            logger.error(f"获取题目失败: {sheet.question_id}")
            self._progress_current += 1
            self._update_progress("error")
            return

        answers, source = self._get_answer(question)
        if answers:
            self._submitted_answer_str[sheet.question_id] = "#@#".join(str(a) for a in answers)
            self._save_answer(sheet.question_id, answers)

        self._progress_current += 1
        self._progress_sources.append(source)
        self._update_progress(source)

        if self._progress_view:
            logger.info(f"题目 {sheet.question_id}: {source}")

    def _finish(self, submit: bool, sheets: list[QuestionSheet]) -> tuple[bool, int, int]:
        """作业直接提交并检查结果；用户跳过时不提交"""
        if self._skipped:
            logger.info("用户跳过当前作业，不提交")
            return False, 0, len(sheets)
        self._submit(submit)
        correct_count, total_count = self._check_results(sheets)
        return correct_count == total_count, correct_count, total_count

    def start(
        self,
        reference_materials: list[dict[str, str]] | None = None,
        submit: bool = True,
    ) -> tuple[bool, int, int]:
        """执行完整作业流程，返回 (是否全对, 正确数, 总题数)

        Args:
            reference_materials: 参考资料（PPT 等）
            submit: 是否提交（作业默认提交）
        """
        return super().start(reference_materials, submit=submit)

"""ai/course.py 补充测试

覆盖 run_course / _run_play_only / _run_homework_only / _collect_completed_ppts /
get_ai_course_list / query_homework 等方法。
"""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from zhs.ai.course import AiCourseManager
from zhs.ai.models import (
    AiCourseInfo,
    ExamInfo,
    KnowledgePoint,
    Resource,
    ResourceDetail,
    Theme,
)
from zhs.config import AIConfig, HomeworkConfig, VideoConfig


@pytest.fixture
def mock_session() -> MagicMock:
    """创建 mock ZhsSession"""
    session = MagicMock()
    session.crypto = MagicMock()
    session.crypto.ai_key = b"hw2fdlwcj4cs1mx7"
    session.crypto.home_key = b"home_key_bytes_16"
    session.crypto.key_bytes = MagicMock(side_effect=lambda name: getattr(session.crypto, name))
    session.urls = MagicMock()
    session.urls.ai = "https://kg-ai-run.zhihuishu.com"
    session.urls.ai_task = "https://kg-run-student.zhihuishu.com"
    session.urls.base = "https://onlineservice-api.zhihuishu.com"
    session.urls.exam = "https://studentexamtest.zhihuishu.com"
    session.speed = 1.25
    return session


@pytest.fixture
def ai_config() -> AIConfig:
    """AI 配置"""
    return AIConfig(api_key="test-key", model="gpt-4o-mini")


@pytest.fixture
def homework_config() -> HomeworkConfig:
    """作业配置"""
    return HomeworkConfig(ai_homework_threshold=90)


@pytest.fixture
def manager(mock_session: MagicMock) -> AiCourseManager:
    """创建 AiCourseManager 实例"""
    return AiCourseManager(mock_session)


class TestGetAiCourseList:
    """get_ai_course_list"""

    def test_returns_list_on_success(self, manager: AiCourseManager) -> None:
        """成功时返回课程列表"""
        mock_data: dict[str, Any] = {
            "rt": [
                {"courseId": 100, "classId": 200, "courseName": "课程1"},
                {"courseId": 101, "classId": 201, "courseName": "课程2"},
            ]
        }
        with patch.object(manager._session, "zhidao_query", return_value=mock_data):
            result = manager.get_ai_course_list()
        assert len(result) == 2
        assert result[0]["courseId"] == 100

    def test_returns_empty_list_when_rt_not_list(self, manager: AiCourseManager) -> None:
        """rt 不是列表时返回空列表"""
        mock_data: dict[str, Any] = {"rt": None}
        with patch.object(manager._session, "zhidao_query", return_value=mock_data):
            result = manager.get_ai_course_list()
        assert result == []

    def test_returns_empty_list_when_rt_missing(self, manager: AiCourseManager) -> None:
        """rt 字段缺失时返回空列表"""
        mock_data: dict[str, Any] = {}
        with patch.object(manager._session, "zhidao_query", return_value=mock_data):
            result = manager.get_ai_course_list()
        assert result == []

    def test_returns_empty_list_on_exception(self, manager: AiCourseManager) -> None:
        """异常时返回空列表"""
        with patch.object(manager._session, "zhidao_query", side_effect=Exception("err")):
            result = manager.get_ai_course_list()
        assert result == []

    def test_uses_home_key_for_encryption(self, manager: AiCourseManager) -> None:
        """使用 home_key 加密"""
        with patch.object(manager._session, "zhidao_query", return_value={"rt": []}) as mock_query:
            manager.get_ai_course_list()
            call_kwargs = mock_query.call_args.kwargs
            assert call_kwargs["key"] == b"home_key_bytes_16"
            assert call_kwargs["ok_code"] == 0


class TestQueryHomework:
    """query_homework"""

    def test_returns_exam_info_on_success(self, manager: AiCourseManager) -> None:
        """成功时返回 ExamInfo"""
        mock_data: dict[str, Any] = {"data": {"examTestId": 1, "paperId": 2, "highMasteryScore": 50}}
        with patch.object(manager, "_ai_query", return_value=mock_data):
            result = manager.query_homework(100, 200, 1)
        assert result is not None
        assert isinstance(result, ExamInfo)
        assert result.exam_test_id == 1
        assert result.paper_id == 2
        assert result.mastery_score == 50

    def test_returns_none_on_exception(self, manager: AiCourseManager) -> None:
        """异常时返回 None"""
        with patch.object(manager, "_ai_query", side_effect=Exception("err")):
            result = manager.query_homework(100, 200, 1)
        assert result is None


class TestCollectCompletedPpts:
    """_collect_completed_ppts"""

    def test_collects_completed_ppt_urls(self, manager: AiCourseManager) -> None:
        """收集已完成 PPT 的 URL"""
        resources = [
            Resource(
                study_status=1,
                resources_detail=ResourceDetail(
                    resources_uid=1,
                    resources_name="PPT1",
                    resources_type=1,
                    resources_distribute_type=4,
                    resources_url="https://example.com/ppt1.pptx",
                ),
            ),
            Resource(
                study_status=0,  # 未完成
                resources_detail=ResourceDetail(
                    resources_uid=2,
                    resources_name="PPT2",
                    resources_type=1,
                    resources_distribute_type=4,
                    resources_url="https://example.com/ppt2.pptx",
                ),
            ),
            Resource(
                study_status=1,  # 已完成但非 PPT
                resources_detail=ResourceDetail(
                    resources_uid=3,
                    resources_name="视频",
                    resources_type=1,
                    resources_distribute_type=3,
                ),
            ),
        ]
        ppts: list[dict[str, str]] = []
        with patch.object(manager, "list_knowledge_resources", return_value=resources):
            manager._collect_completed_ppts(100, 200, 1, ppts)

        assert len(ppts) == 1
        assert ppts[0]["name"] == "PPT1"
        assert ppts[0]["url"] == "https://example.com/ppt1.pptx"

    def test_exception_does_not_raise(self, manager: AiCourseManager) -> None:
        """异常时不抛出"""
        ppts: list[dict[str, str]] = []
        with patch.object(manager, "list_knowledge_resources", side_effect=Exception("err")):
            # 不应抛异常
            manager._collect_completed_ppts(100, 200, 1, ppts)
        assert ppts == []

    def test_skips_ppt_without_url(self, manager: AiCourseManager) -> None:
        """跳过无 URL 的 PPT"""
        resources = [
            Resource(
                study_status=1,
                resources_detail=ResourceDetail(
                    resources_uid=1,
                    resources_name="PPT无URL",
                    resources_type=1,
                    resources_distribute_type=4,
                    resources_url="",  # 空 URL
                ),
            ),
        ]
        ppts: list[dict[str, str]] = []
        with patch.object(manager, "list_knowledge_resources", return_value=resources):
            manager._collect_completed_ppts(100, 200, 1, ppts)

        assert ppts == []


class TestRunPlayOnly:
    """_run_play_only"""

    def test_skips_completed_knowledge(self, manager: AiCourseManager) -> None:
        """跳过已完成的知识点（study_progress >= 101）"""
        knowledge = KnowledgePoint(knowledge_id=1, knowledge_name="已完成", study_progress=101)
        video_player = MagicMock()

        with patch.object(manager, "list_knowledge_resources") as mock_list:
            manager._run_play_only(100, 200, knowledge, video_player, learn_optional=False)

        # 不应调用 list_knowledge_resources
        mock_list.assert_not_called()

    def test_processes_incomplete_knowledge(self, manager: AiCourseManager) -> None:
        """处理未完成的知识点"""
        knowledge = KnowledgePoint(knowledge_id=1, knowledge_name="未完成", study_progress=50)
        video_player = MagicMock()
        resources = [
            Resource(
                study_status=0,
                resources_detail=ResourceDetail(
                    resources_uid=1,
                    resources_name="文本",
                    resources_type=2,
                    resources_distribute_type=1,
                ),
            )
        ]
        with (
            patch.object(manager, "list_knowledge_resources", return_value=resources),
            patch.object(manager, "complete_resource") as mock_complete,
            patch("zhs.ai.course.time.sleep"),  # 跳过随机延迟
        ):
            manager._run_play_only(100, 200, knowledge, video_player, learn_optional=False)

        mock_complete.assert_called_once()

    def test_list_resources_exception_does_not_raise(self, manager: AiCourseManager) -> None:
        """list_knowledge_resources 异常不抛出"""
        knowledge = KnowledgePoint(knowledge_id=1, knowledge_name="测试", study_progress=0)
        video_player = MagicMock()
        with patch.object(manager, "list_knowledge_resources", side_effect=Exception("err")):
            # 不应抛异常
            manager._run_play_only(100, 200, knowledge, video_player, learn_optional=False)

    def test_process_resource_exception_continues(self, manager: AiCourseManager) -> None:
        """单个资源处理异常不中断"""
        knowledge = KnowledgePoint(knowledge_id=1, knowledge_name="测试", study_progress=0)
        video_player = MagicMock()
        resources = [
            Resource(
                study_status=0,
                resources_detail=ResourceDetail(
                    resources_uid=1,
                    resources_name="资源1",
                    resources_type=2,
                    resources_distribute_type=1,
                ),
            ),
            Resource(
                study_status=0,
                resources_detail=ResourceDetail(
                    resources_uid=2,
                    resources_name="资源2",
                    resources_type=2,
                    resources_distribute_type=1,
                ),
            ),
        ]
        with (
            patch.object(manager, "list_knowledge_resources", return_value=resources),
            patch.object(manager, "_process_resource", side_effect=[Exception("err"), None]) as mock_process,
            patch("zhs.ai.course.time.sleep"),
        ):
            manager._run_play_only(100, 200, knowledge, video_player, learn_optional=False)
            # 第二个资源仍应被处理（_process_resource 被调用 2 次）
            assert mock_process.call_count == 2


class TestRunHomeworkOnly:
    """_run_homework_only"""

    def test_skips_when_homework_done(self, manager: AiCourseManager) -> None:
        """作业已达标时跳过"""
        knowledge = KnowledgePoint(knowledge_id=1, knowledge_name="测试", study_progress=0)
        theme = Theme(theme_name="主题1")
        course_info = AiCourseInfo(course_name="课程1")
        homework_config = HomeworkConfig(ai_homework_threshold=90)

        # exam.mastery_score >= 90 → 跳过
        exam = ExamInfo(exam_test_id=1, paper_id=2, mastery_score=95)
        with patch.object(manager, "query_homework", return_value=exam):
            manager._run_homework_only(100, 200, knowledge, theme, course_info, AIConfig(api_key="k"), homework_config)

    def test_skips_when_no_exam(self, manager: AiCourseManager) -> None:
        """无作业时跳过"""
        knowledge = KnowledgePoint(knowledge_id=1, knowledge_name="测试", study_progress=0)
        theme = Theme(theme_name="主题1")
        course_info = AiCourseInfo(course_name="课程1")
        homework_config = HomeworkConfig(ai_homework_threshold=90)

        with patch.object(manager, "query_homework", return_value=None):
            manager._run_homework_only(100, 200, knowledge, theme, course_info, AIConfig(api_key="k"), homework_config)

    def test_skips_when_no_paper_id(self, manager: AiCourseManager) -> None:
        """无 paper_id 时跳过"""
        knowledge = KnowledgePoint(knowledge_id=1, knowledge_name="测试", study_progress=0)
        theme = Theme(theme_name="主题1")
        course_info = AiCourseInfo(course_name="课程1")
        homework_config = HomeworkConfig(ai_homework_threshold=90)

        exam = ExamInfo(exam_test_id=1, paper_id=0, mastery_score=50)
        with patch.object(manager, "query_homework", return_value=exam):
            manager._run_homework_only(100, 200, knowledge, theme, course_info, AIConfig(api_key="k"), homework_config)

    def test_stops_after_consecutive_no_improvement(self, manager: AiCourseManager) -> None:
        """分数连续无提升达到上限后停止重试（避免 30~阈值区间无限循环）"""
        knowledge = KnowledgePoint(knowledge_id=1, knowledge_name="测试", study_progress=0)
        theme = Theme(theme_name="主题1")
        course_info = AiCourseInfo(course_name="课程1")
        homework_config = HomeworkConfig(
            ai_homework_threshold=90,
            ai_homework_max_stall=3,
            ai_homework_max_attempts=100,
        )
        stuck = ExamInfo(exam_test_id=1, paper_id=2, mastery_score=80)

        mock_ctx = MagicMock()
        mock_ctx.skipped = False
        mock_ctx.used_reference = False
        mock_ctx.start.return_value = (False, 5, 6)
        with (
            patch.object(manager, "query_homework", return_value=stuck),
            patch.object(manager, "_collect_completed_ppts"),
            patch("zhs.ai.homework.HomeworkCtx", return_value=mock_ctx),
            patch("zhs.ai.course.time.sleep"),
        ):
            manager._run_homework_only(100, 200, knowledge, theme, course_info, AIConfig(api_key="k"), homework_config)

        # 首次 + 连续 3 次无提升 = 3 次实际作答（停滞计数在作答后判定）
        assert mock_ctx.start.call_count == 3

    def test_passes_reference_fetcher_to_homework_ctx(self, manager: AiCourseManager) -> None:
        """标准答案查询函数传给 HomeworkCtx，按题目 ID 查询"""
        knowledge = KnowledgePoint(knowledge_id=1, knowledge_name="测试", study_progress=0)
        theme = Theme(theme_name="主题1")
        course_info = AiCourseInfo(course_name="课程1")
        homework_config = HomeworkConfig(ai_homework_threshold=90)
        stuck = ExamInfo(exam_test_id=1, paper_id=2, mastery_score=50)

        mock_ctx = MagicMock()
        mock_ctx.skipped = True
        mock_ctx.start.return_value = (False, 0, 1)
        with (
            patch.object(manager, "query_homework", return_value=stuck),
            patch.object(manager, "_collect_completed_ppts"),
            patch.object(manager, "_next_reference_candidate", return_value="标准答案") as mock_result,
            patch("zhs.ai.homework.HomeworkCtx", return_value=mock_ctx) as mock_cls,
            patch("zhs.ai.course.SkipInput") as mock_skip_cls,
            patch("zhs.ai.course.time.sleep"),
        ):
            mock_skip_cls.return_value.enabled = False
            manager._run_homework_only(100, 200, knowledge, theme, course_info, AIConfig(api_key="k"), homework_config)

            fetcher = mock_cls.call_args.kwargs["reference_fetcher"]
            assert fetcher(1010786456) == "标准答案"
            mock_result.assert_called_once_with(100, 200, 1, 1, 1010786456)

            with patch.object(manager, "get_question_option_contents", return_value=["选项A"]) as mock_options:
                options_fetcher = mock_cls.call_args.kwargs["reference_options_fetcher"]
                assert options_fetcher(2) == ["选项A"]
                mock_options.assert_called_once_with(100, 200, 1, 1, 2)

    def test_stops_after_user_skips(self, manager: AiCourseManager) -> None:
        """用户按 p 跳过后不再重试，直接进入下一个作业"""
        knowledge = KnowledgePoint(knowledge_id=1, knowledge_name="测试", study_progress=0)
        theme = Theme(theme_name="主题1")
        course_info = AiCourseInfo(course_name="课程1")
        homework_config = HomeworkConfig(ai_homework_threshold=90, ai_homework_max_attempts=10)
        stuck = ExamInfo(exam_test_id=1, paper_id=2, mastery_score=50)

        mock_ctx = MagicMock()
        mock_ctx.skipped = True
        mock_ctx.start.return_value = (False, 0, 6)
        with (
            patch.object(manager, "query_homework", return_value=stuck),
            patch.object(manager, "_collect_completed_ppts"),
            patch("zhs.ai.homework.HomeworkCtx", return_value=mock_ctx),
            patch("zhs.ai.course.SkipInput") as mock_skip_cls,
            patch("zhs.ai.course.time.sleep"),
        ):
            mock_skip_cls.return_value.enabled = False
            manager._run_homework_only(100, 200, knowledge, theme, course_info, AIConfig(api_key="k"), homework_config)

        assert mock_ctx.start.call_count == 1
        mock_skip_cls.return_value.start.assert_called_once()
        mock_skip_cls.return_value.stop.assert_called_once()

    def test_stops_at_max_attempts(self, manager: AiCourseManager) -> None:
        """即使分数持续小幅提升，也受最大尝试次数约束"""
        knowledge = KnowledgePoint(knowledge_id=1, knowledge_name="测试", study_progress=0)
        theme = Theme(theme_name="主题1")
        course_info = AiCourseInfo(course_name="课程1")
        homework_config = HomeworkConfig(
            ai_homework_threshold=100,
            ai_homework_max_stall=0,
            ai_homework_max_attempts=3,
        )
        scores = [0, 10, 20, 30, 40, 50]
        exams = [ExamInfo(exam_test_id=1, paper_id=2, mastery_score=s) for s in scores]

        mock_ctx = MagicMock()
        mock_ctx.skipped = False
        mock_ctx.used_reference = False
        mock_ctx.start.return_value = (False, 5, 6)
        with (
            patch.object(manager, "query_homework", side_effect=exams),
            patch.object(manager, "_collect_completed_ppts"),
            patch("zhs.ai.homework.HomeworkCtx", return_value=mock_ctx),
            patch("zhs.ai.course.time.sleep"),
        ):
            manager._run_homework_only(100, 200, knowledge, theme, course_info, AIConfig(api_key="k"), homework_config)

        assert mock_ctx.start.call_count == 3


class TestGetQuestionAnswerCandidates:
    """get_question_answer_candidates / 候选轮换"""

    def test_builds_candidates_from_latex_result(self, manager: AiCourseManager) -> None:
        """含 $$ 的填空题按 LaTeX 变体生成候选，单 $ 优先"""
        payload = {
            "code": 200,
            "data": {
                "questionList": [
                    {
                        "id": "1010786456",
                        "result": "相对运动速度 $$\\vec{v}_r$$",
                        "optionVos": [
                            {
                                "content": '<div>相对运动速度 <img src="https://x/a.png"></div>',
                                "isCorrect": 1,
                            }
                        ],
                    }
                ]
            },
        }
        with patch.object(manager, "_ai_query", return_value=payload) as mock_query:
            candidates = manager.get_question_answer_candidates(100, 200, 300, 400, 1010786456)

        assert candidates[0] == "相对运动速度 $\\vec{v}_r$"
        assert "相对运动速度 \\vec{v}_r" in candidates
        assert "相对运动速度 v_r" in candidates
        assert "相对运动速度 $$\\vec{v}_r$$" in candidates
        assert "相对运动速度" in candidates
        assert "https://x/a.png" in candidates
        assert "mastery-statistics/get-test-question" in mock_query.call_args.args[0]
        assert mock_query.call_args.args[1]["questionId"] == 1010786456

    def test_svg_content_prioritized(self, manager: AiCourseManager) -> None:
        """题库 MathJax SVG 内容优先（实测该形式能通过自动批阅）"""
        svg = (
            '<div><span class="math-block "><mjx-container><svg data-svg-latex="x"></svg></mjx-container></span></div>'
        )
        payload = {
            "code": 200,
            "data": {
                "questionList": [
                    {
                        "id": 1010786456,
                        "result": "相对运动速度 $$\\vec{v}_r$$",
                        "optionVos": [{"content": svg, "isCorrect": 1}],
                    }
                ]
            },
        }
        with patch.object(manager, "_ai_query", return_value=payload):
            candidates = manager.get_question_answer_candidates(100, 200, 300, 400, 1010786456)
        assert candidates[0] == svg
        assert "相对运动速度 $\\vec{v}_r$" in candidates

    def test_plain_result_first(self, manager: AiCourseManager) -> None:
        """纯文本答案不需要变体，result 直接排第一"""
        payload = {"code": 200, "data": {"questionList": [{"id": 456, "result": "力偶", "optionVos": []}]}}
        with patch.object(manager, "_ai_query", return_value=payload):
            candidates = manager.get_question_answer_candidates(100, 200, 300, 400, 456)
        assert candidates[0] == "力偶"

    def test_other_question_returns_empty(self, manager: AiCourseManager) -> None:
        """返回的不是请求的题目时没有候选"""
        payload = {"code": 200, "data": {"questionList": [{"id": 456, "result": "别的答案"}]}}
        with patch.object(manager, "_ai_query", return_value=payload):
            assert manager.get_question_answer_candidates(100, 200, 300, 400, 1010786456) == []

    def test_api_error_returns_empty(self, manager: AiCourseManager) -> None:
        """接口异常时返回空列表，不影响作业流程"""
        with patch.object(manager, "_ai_query", side_effect=Exception("network")):
            assert manager.get_question_answer_candidates(100, 200, 300, 400, 1010786456) == []

    def test_malformed_response_returns_empty(self, manager: AiCourseManager) -> None:
        """响应结构异常时返回空列表"""
        with patch.object(manager, "_ai_query", return_value=MagicMock()):
            assert manager.get_question_answer_candidates(100, 200, 300, 400, 1010786456) == []
        with patch.object(manager, "_ai_query", return_value={"data": None}):
            assert manager.get_question_answer_candidates(100, 200, 300, 400, 1010786456) == []

    def test_rotates_candidates(self, manager: AiCourseManager) -> None:
        """每次重试返回下一个候选，用完后返回 None"""
        with patch.object(manager, "get_question_answer_candidates", return_value=["候选1", "候选2"]):
            assert manager._next_reference_candidate(100, 200, 300, 400, 1) == "候选1"
            assert manager._next_reference_candidate(100, 200, 300, 400, 1) == "候选2"
            assert manager._next_reference_candidate(100, 200, 300, 400, 1) is None


class TestGetQuestionOptionContents:
    """get_question_option_contents"""

    def test_returns_correct_option_contents(self, manager: AiCourseManager) -> None:
        """返回 isCorrect=1 的选项内容"""
        payload = {
            "code": 200,
            "data": {
                "questionList": [
                    {
                        "id": 930916063,
                        "optionVos": [
                            {"content": "<p>A</p>", "isCorrect": 1},
                            {"content": "<p>B</p>", "isCorrect": 0},
                            {"content": "<p>C</p>", "isCorrect": 1},
                        ],
                    }
                ]
            },
        }
        with patch.object(manager, "_ai_query", return_value=payload) as mock_query:
            contents = manager.get_question_option_contents(100, 200, 300, 400, 930916063)

        assert contents == ["<p>A</p>", "<p>C</p>"]
        assert mock_query.call_args.args[1]["questionId"] == 930916063

        # 同一题第二次直接命中内存缓存
        manager.get_question_option_contents(100, 200, 300, 400, 930916063)
        assert mock_query.call_count == 1

    def test_other_question_returns_empty(self, manager: AiCourseManager) -> None:
        payload = {"code": 200, "data": {"questionList": [{"id": 1, "optionVos": [{"content": "A", "isCorrect": 1}]}]}}
        with patch.object(manager, "_ai_query", return_value=payload):
            assert manager.get_question_option_contents(100, 200, 300, 400, 930916063) == []

    def test_api_error_returns_empty(self, manager: AiCourseManager) -> None:
        with patch.object(manager, "_ai_query", side_effect=Exception("network")):
            assert manager.get_question_option_contents(100, 200, 300, 400, 930916063) == []

    def test_malformed_response_returns_empty(self, manager: AiCourseManager) -> None:
        with patch.object(manager, "_ai_query", return_value=MagicMock()):
            assert manager.get_question_option_contents(100, 200, 300, 400, 930916063) == []


class TestRunCourse:
    """run_course"""

    def test_play_mode_calls_run_play_only(self, manager: AiCourseManager) -> None:
        """play 模式（no_homework=True）调用 _run_play_only"""
        course_info = AiCourseInfo(
            course_name="测试课程",
            cake_theme_list=[
                Theme(
                    theme_name="主题1",
                    knowledge_list=[KnowledgePoint(knowledge_id=1, knowledge_name="知识点1", study_progress=0)],
                )
            ],
        )
        with (
            patch.object(manager, "get_knowledge_points", return_value=course_info),
            patch.object(manager, "_run_play_only") as mock_play,
            patch("zhs.ai.course.time.sleep"),
            patch("zhs.ai.course.AiVideoPlayer"),
        ):
            manager.run_course(
                100,
                200,
                AIConfig(api_key="k"),
                HomeworkConfig(),
                video_config=VideoConfig(),
                no_homework=True,
            )

        mock_play.assert_called_once()

    def test_homework_mode_calls_run_homework_only(self, manager: AiCourseManager) -> None:
        """homework 模式（no_homework=False）调用 _run_homework_only"""
        course_info = AiCourseInfo(
            course_name="测试课程",
            cake_theme_list=[
                Theme(
                    theme_name="主题1",
                    knowledge_list=[KnowledgePoint(knowledge_id=1, knowledge_name="知识点1", study_progress=0)],
                )
            ],
        )
        with (
            patch.object(manager, "get_knowledge_points", return_value=course_info),
            patch.object(manager, "_run_homework_only") as mock_hw,
            patch("zhs.ai.course.time.sleep"),
            patch("zhs.ai.course.AiVideoPlayer"),
        ):
            manager.run_course(
                100,
                200,
                AIConfig(api_key="k"),
                HomeworkConfig(),
                video_config=VideoConfig(),
                no_homework=False,
            )

        mock_hw.assert_called_once()

    def test_empty_theme_list(self, manager: AiCourseManager) -> None:
        """空主题列表不抛异常"""
        course_info = AiCourseInfo(course_name="空课程", cake_theme_list=[])
        with (
            patch.object(manager, "get_knowledge_points", return_value=course_info),
            patch("zhs.ai.course.AiVideoPlayer"),
        ):
            manager.run_course(
                100,
                200,
                AIConfig(api_key="k"),
                HomeworkConfig(),
                video_config=VideoConfig(),
                no_homework=True,
            )

    def test_default_video_config_when_none(self, manager: AiCourseManager) -> None:
        """video_config=None 时使用默认 VideoConfig"""
        course_info = AiCourseInfo(course_name="测试", cake_theme_list=[])
        with (
            patch.object(manager, "get_knowledge_points", return_value=course_info),
            patch("zhs.ai.course.AiVideoPlayer"),
        ):
            manager.run_course(
                100,
                200,
                AIConfig(api_key="k"),
                HomeworkConfig(),
                video_config=None,
                no_homework=True,
            )

    def test_node_uid_direct_mode_play(self, manager: AiCourseManager) -> None:
        """node_uid 直接模式（play）：只处理匹配的知识点，跳过其他"""
        course_info = AiCourseInfo(
            course_name="测试课程",
            cake_theme_list=[
                Theme(
                    theme_name="主题1",
                    knowledge_list=[
                        KnowledgePoint(knowledge_id=10, knowledge_name="知识点A", study_progress=0),
                        KnowledgePoint(knowledge_id=20, knowledge_name="知识点B", study_progress=0),
                    ],
                )
            ],
        )
        with (
            patch.object(manager, "get_knowledge_points", return_value=course_info),
            patch.object(manager, "_run_play_only") as mock_play,
            patch("zhs.ai.course.time.sleep"),
            patch("zhs.ai.course.AiVideoPlayer"),
        ):
            manager.run_course(
                100,
                200,
                AIConfig(api_key="k"),
                HomeworkConfig(),
                video_config=VideoConfig(),
                no_homework=True,
                node_uid=20,
            )

        # 只调用了 1 次（知识点B），知识点A 被跳过
        mock_play.assert_called_once()
        # 验证传入的是知识点B（第 3 个位置参数）
        args = mock_play.call_args
        assert args[0][2].knowledge_id == 20

    def test_node_uid_direct_mode_homework(self, manager: AiCourseManager) -> None:
        """node_uid 直接模式（homework）：只处理匹配的知识点"""
        course_info = AiCourseInfo(
            course_name="测试课程",
            cake_theme_list=[
                Theme(
                    theme_name="主题1",
                    knowledge_list=[
                        KnowledgePoint(knowledge_id=10, knowledge_name="知识点A", study_progress=0),
                        KnowledgePoint(knowledge_id=20, knowledge_name="知识点B", study_progress=0),
                    ],
                )
            ],
        )
        with (
            patch.object(manager, "get_knowledge_points", return_value=course_info),
            patch.object(manager, "_run_homework_only") as mock_hw,
            patch("zhs.ai.course.time.sleep"),
            patch("zhs.ai.course.AiVideoPlayer"),
        ):
            manager.run_course(
                100,
                200,
                AIConfig(api_key="k"),
                HomeworkConfig(),
                video_config=VideoConfig(),
                no_homework=False,
                node_uid=10,
            )

        mock_hw.assert_called_once()
        args = mock_hw.call_args
        assert args[0][2].knowledge_id == 10

    def test_node_uid_not_found_fallback_scan(self, manager: AiCourseManager) -> None:
        """node_uid 不匹配任何知识点时回退扫描模式（全刷）"""
        course_info = AiCourseInfo(
            course_name="测试课程",
            cake_theme_list=[
                Theme(
                    theme_name="主题1",
                    knowledge_list=[
                        KnowledgePoint(knowledge_id=10, knowledge_name="知识点A", study_progress=0),
                        KnowledgePoint(knowledge_id=20, knowledge_name="知识点B", study_progress=0),
                    ],
                )
            ],
        )
        with (
            patch.object(manager, "get_knowledge_points", return_value=course_info),
            patch.object(manager, "_run_play_only") as mock_play,
            patch("zhs.ai.course.time.sleep"),
            patch("zhs.ai.course.AiVideoPlayer"),
        ):
            manager.run_course(
                100,
                200,
                AIConfig(api_key="k"),
                HomeworkConfig(),
                video_config=VideoConfig(),
                no_homework=True,
                node_uid=999,  # 不存在的知识点
            )

        # 回退扫描：处理所有 2 个知识点
        assert mock_play.call_count == 2

    def test_node_uid_none_scans_all(self, manager: AiCourseManager) -> None:
        """node_uid=None 时扫描所有知识点"""
        course_info = AiCourseInfo(
            course_name="测试课程",
            cake_theme_list=[
                Theme(
                    theme_name="主题1",
                    knowledge_list=[
                        KnowledgePoint(knowledge_id=10, knowledge_name="知识点A", study_progress=0),
                        KnowledgePoint(knowledge_id=20, knowledge_name="知识点B", study_progress=0),
                        KnowledgePoint(knowledge_id=30, knowledge_name="知识点C", study_progress=0),
                    ],
                )
            ],
        )
        with (
            patch.object(manager, "get_knowledge_points", return_value=course_info),
            patch.object(manager, "_run_play_only") as mock_play,
            patch("zhs.ai.course.time.sleep"),
            patch("zhs.ai.course.AiVideoPlayer"),
        ):
            manager.run_course(
                100,
                200,
                AIConfig(api_key="k"),
                HomeworkConfig(),
                video_config=VideoConfig(),
                no_homework=True,
                node_uid=None,
            )

        assert mock_play.call_count == 3


class TestOptionalResourceFiltering:
    """选学资源过滤"""

    def test_optional_resource_skipped_by_default(self, manager: AiCourseManager) -> None:
        """默认跳过选学资源（resourcesSyncType != 1）"""
        detail = ResourceDetail(
            resources_uid=1,
            resources_name="选学",
            resources_type=2,
            resources_distribute_type=1,
            resources_sync_type=2,  # 非必修
        )
        resource = Resource(study_status=0, resources_detail=detail)
        video_player = MagicMock()
        with patch.object(manager, "complete_resource") as mock_complete:
            manager._process_resource(100, 200, 1, resource, video_player, learn_optional=False)
        mock_complete.assert_not_called()

    def test_optional_resource_processed_when_learn_optional(self, manager: AiCourseManager) -> None:
        """learn_optional=True 时处理选学资源"""
        detail = ResourceDetail(
            resources_uid=1,
            resources_name="选学",
            resources_type=2,
            resources_distribute_type=1,
            resources_sync_type=2,  # 非必修
        )
        resource = Resource(study_status=0, resources_detail=detail)
        video_player = MagicMock()
        with (
            patch.object(manager, "complete_resource") as mock_complete,
            patch("zhs.ai.course.time.sleep"),
        ):
            manager._process_resource(100, 200, 1, resource, video_player, learn_optional=True)
        mock_complete.assert_called_once()

    def test_optional_completed_ppt_still_collected(self, manager: AiCourseManager) -> None:
        """已完成的选学 PPT 仍收集 URL（即使 learn_optional=False）"""
        detail = ResourceDetail(
            resources_uid=1,
            resources_name="选学PPT",
            resources_type=1,
            resources_distribute_type=4,
            resources_sync_type=2,  # 非必修
            resources_url="https://example.com/ppt.pptx",
        )
        resource = Resource(study_status=1, resources_detail=detail)
        ppts: list[dict[str, str]] = []
        video_player = MagicMock()
        with patch.object(manager, "complete_resource") as mock_complete:
            manager._process_resource(100, 200, 1, resource, video_player, ppts=ppts, learn_optional=False)
        mock_complete.assert_not_called()
        assert len(ppts) == 1
        assert ppts[0]["url"] == "https://example.com/ppt.pptx"

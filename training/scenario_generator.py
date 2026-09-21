# -*- coding: utf-8 -*-
"""
Gemini-based per-participant multi-turn scenario generator.

Input:
  participants/<pid>/phase1/conversation_log.json   (actual 10 Phase 1 answer turns)
  config/noise.json                                  (shared noise pool across participants)
  GEMINI_API_KEY (.env or environment variable)

Output:
  memory_data/scenarios/<pid>_scenarios.json  (10 × 20-turn scenarios)

Design principles:
  - All three conditions (in_context/retrieval/parametric) use this artifact as a shared source.
  - Gemini may only reuse words/figures from the Phase 1 answers verbatim. No new facts allowed.
  - Phase 2 small talk (turns 6–15) is naturally blended in, referencing topics from the noise pool.
  - Phase 3 (turns 16–20) is the callback segment where the AI brings up Phase 1 facts again.
  - Each scenario rotates the required cover fields to evenly expose the 10 profile fields.
  - User and assistant turns strictly alternate, fixed at 20 turns total.

Standalone run:
  python training/scenario_generator.py -p P001 -n 10

Integrated run (called automatically from build_memory.py):
  python build_memory.py -p P001
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

# Prevent a race condition when accumulating Gemini usage
_USAGE_LOCK = threading.Lock()


# ─── Constants ────────────────────────────────────────────────────
_QUESTION_FIELD_MAP: list[tuple[str, str]] = [
    ("성함", "이름"),
    ("나이", "나이"),
    ("일을 하고", "직업/전공"),
    ("살고 계세요", "거주지"),
    ("취미", "취미"),
    ("좋아하는 음식", "좋아하는 음식"),
    ("운동", "운동 습관"),
    ("여행", "최근 여행"),
    ("영상이나 콘텐츠", "관심 콘텐츠"),
    ("관심 있는 주제", "관심 주제"),
]

_STT_NOISE_RE = re.compile(r"\([^)]*\)")
_POLITE_TAIL_RE = re.compile(
    r"(입니다|습니다|이에요|예요|이야|같습니다|같아요|해요|해|어요|지요|죠)\.?\??$"
)
_JOSA_TAIL_RE = re.compile(
    r"(이|가|은|는|을|를|도|만|부터|까지|에서|에게|에|의|과|와|로|으로)$"
)
_CONNECTIVE_TAIL_RE = re.compile(
    r"(이고|이나|하며|하고|한다|했다|하니|이라|이라고|같습니다|같아요|"
    r"입니다|습니다|있습니다|합니다|합니|이라서|라서|이지만|지만)$"
)

DEFAULT_MODEL = "gemini-3-flash-preview"
DEFAULT_NUM_SCENARIOS = 10
TURNS_PER_SCENARIO = 20
MAX_RETRIES_PER_SCENARIO = 3

# Per-scenario "this time you must cover these fields" rotation.
# 10 scenarios × 4 fields = 40 slots. Name is included every time (identity anchor),
# and the remaining 9 fields are distributed so each appears 3–4 times.
_FIELD_ROTATION: list[list[str]] = [
    ["이름", "나이",       "직업/전공",  "거주지"],        # 1: core identity
    ["이름", "취미",       "좋아하는 음식", "최근 여행"],   # 2: preferences/experience
    ["이름", "운동 습관",   "관심 콘텐츠", "관심 주제"],    # 3: activity/interests
    ["이름", "나이",       "거주지",     "취미"],         # 4
    ["이름", "직업/전공",   "좋아하는 음식", "최근 여행"],   # 5
    ["이름", "운동 습관",   "관심 콘텐츠", "나이"],        # 6
    ["이름", "거주지",     "관심 주제",   "취미"],        # 7
    ["이름", "직업/전공",   "운동 습관",   "좋아하는 음식"], # 8
    ["이름", "거주지",     "관심 콘텐츠", "최근 여행"],     # 9
    ["이름", "나이",       "관심 주제",   "운동 습관"],    # 10
]


# ─── Utilities ────────────────────────────────────────────────────

def _clean_user_answer(text: str) -> str:
    return _STT_NOISE_RE.sub("", text).strip()


def extract_profile(phase1_log: list[dict]) -> dict[str, str]:
    """Map the 10 turns of Phase 1 conversation_log.json into a profile dict."""
    profile: dict[str, str] = {}
    for turn in phase1_log:
        agent_q = turn.get("agent_text", "") or ""
        user_a = _clean_user_answer(turn.get("user_text", "") or "")
        if not user_a:
            continue
        for keyword, field in _QUESTION_FIELD_MAP:
            if keyword in agent_q and field not in profile:
                profile[field] = user_a
                break
    return profile


def load_noise_pool(noise_path: Path) -> list[dict]:
    if not noise_path.exists():
        return []
    with noise_path.open(encoding="utf-8") as f:
        return json.load(f)


def _extract_key_tokens(value: str) -> list[str]:
    """Extract candidate validation tokens from a profile value (strip particles/connective endings/polite endings)."""
    if not value:
        return []
    core = _POLITE_TAIL_RE.sub("", value).strip(". ?!")
    parts = re.split(r"[\s,·/]+", core)
    tokens: list[str] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        for _ in range(2):
            p = _JOSA_TAIL_RE.sub("", p)
            p = _CONNECTIVE_TAIL_RE.sub("", p)
        if len(p) >= 2 and p not in tokens:
            tokens.append(p)
    return tokens


# ─── Gemini calls ─────────────────────────────────────────────────

def _required_fields_for(scenario_id: int, profile: dict[str, str]) -> list[str]:
    idx = (scenario_id - 1) % len(_FIELD_ROTATION)
    required = _FIELD_ROTATION[idx]
    return [f for f in required if f in profile]


def _opening_mode_for(scenario_id: int) -> tuple[str, str]:
    """Scenario id → (mode label, prompt description).

    Only 3 of 10 (1–3) are first meetings; the other 7 (4–10) are already mid-conversation or reunions.
    Diversifies scenarios so not every conversation opens with "처음 뵙겠습니다".
    """
    # Korean by design — experiment stimulus
    sid = ((scenario_id - 1) % 10) + 1
    if sid <= 3:
        return (
            "first_meeting",
            "첫 만남 — 서로를 아직 잘 모르는 초대면. 가벼운 자기소개가 섞여도 자연스러움.",
        )
    elif sid <= 7:
        return (
            "continuation",
            "이미 여러 번 대화한 사이 — 익숙한 친구에게 말 걸 듯 자연스럽게 중간부터 시작. "
            "'처음 뵙겠습니다' / '만나서 반가워요' / '안녕하세요 저는~' 같은 초대면 인사말 금지.",
        )
    else:
        return (
            "re_meeting",
            "며칠 만에 다시 만난 사이 — '오랜만이에요', '그동안 어떻게 지냈어요?' 처럼 "
            "공백 후 재회하는 분위기. 첫 만남 인사말 금지.",
        )


def _build_prompt(
    profile: dict[str, str],
    noise_pool: list[dict],
    scenario_id: int,
    required_fields: list[str],
) -> str:
    profile_block = json.dumps(profile, ensure_ascii=False, indent=2)
    # New noise schema: {topic, context_hints, response_strategies}
    noise_topics = "\n".join(
        f"- 주제: {n['topic']} "
        f"(상황 힌트: {', '.join(n.get('context_hints', []))} / "
        f"대응 전략: {', '.join(n.get('response_strategies', []))})"
        for n in noise_pool[:15]
    )
    required_block = "\n".join(
        f"  - {f}: {profile.get(f, '')!r}" for f in required_fields
    )
    opening_label, opening_desc = _opening_mode_for(scenario_id)

    # Korean by design — experiment stimulus
    return f"""당신은 HCI 실험용 멀티턴 대화 데이터 생성기입니다.
제공된 실제 참가자 프로필과 노이즈 풀을 기반으로 자연스러운 20턴 대화 시나리오 1개를 생성하세요.

[실제 참가자 프로필 — 이 값들만 사용자에 대해 참이라고 가정]
{profile_block}

[Phase 2 잡담 주제 풀 (주제 + 상황 힌트 + 대응 전략)]
{noise_topics}

[이번 시나리오의 필수 커버 필드 — 반드시 Phase 1 또는 Phase 3 에서 언급]
{required_block}

[이번 시나리오 오프닝 모드]
{opening_label} — {opening_desc}

[사용자 턴 내용 제약 — 최우선 규칙, 위반 시 시나리오 폐기]
사용자 턴에 담을 수 있는 내용은 **다음 세 유형으로 엄격히 제한** 합니다.
그 외의 개인 사실·선호·경험·고유명사를 사용자 입으로 절대 말하게 하지 마세요.

(A) Phase 1 프로필 값
    위 프로필 10필드에 있는 내용만. 말투·어미는 자유롭게 바꿔도 되지만
    **내용의 고유명사·대상·평가는 그대로 유지**.
    예: "게임 중에서 롤을 조금 많이 하는 것 같습니다"
        → "롤 많이 해요" / "주로 롤 함" / "롤 자주 해" OK
        → "나 미드 주로 가" (포지션 신규) ❌
        → "롤이랑 오버워치 해" (게임 신규) ❌
    예: "부산 여행이 재미있었습니다"
        → "부산 다녀왔어요" / "부산 갔다 왔어" OK
        → "해운대에서 회 먹었어" (지명·메뉴 신규) ❌

(B) 노이즈 풀 topic 의 **상태 표현만**
    위 잡담 topic 이 정의한 감각·상황 상태를 짧게 말하기.
    **선호·의견·과거 행동 금지** — topic 은 "배고픔 상태" 이지 "배고플 때 뭘 먹는지 취향" 이 아님.
    예: "아 배고파" / "졸려" / "날씨 좋다" / "허리 뻐근하네" / "핸드폰 배터리 없네" OK
        → "매운 거 좋아해" (맛 선호 신규) ❌
        → "조용한 카페가 낫지" (환경 선호 신규) ❌
        → "친구들이랑 시켜 먹어" (생활 패턴·관계 신규) ❌
        → "아까 커피 마셨더니 속 쓰려" (과거 행동 신규) ❌

(C) 메타 반응·되묻기
    AI 발화에 대한 짧은 리액션이나 되묻기.
    "너는?" / "넌 어때?" / "왜?" / "진짜?" / "그래요?" / "그렇구나" / "ㅋㅋ" / "맞아요" 수준.
    여기서도 새 사실·선호 금지.

[사용자 턴에 절대 등장 금지 — 체크리스트]
- Phase 1 프로필에 없는 **고유명사 일체**:
  지명(해운대·강남·서울 등), 메뉴(회·국밥·커피 등), 게임 포지션·캐릭터(미드·정글·ADC 등),
  브랜드, 인물명, 학교명 등.
- **선호·호오 표현으로 Phase 1 외 대상을 평가**:
  "~좋아해", "~가 낫지", "~이 최고야", "~싫어" — 평가 대상이 Phase 1 에 없으면 금지.
- **Phase 1 에 없는 과거 경험 세부**:
  "아까 ~했어", "어제 ~갔어", "지난번에 ~먹었어", "예전에 ~했거든" 형태 금지.
- **가족·친구·타인 관계 정보**:
  "친구들이랑 ~", "엄마가 ~", "선배가 ~" — 관계 정보 자체가 Phase 1 에 없으면 금지.
- **신체·건강 세부 사실**:
  "허리 디스크 있어", "알레르기 있어" — 상태 표현을 넘는 병력·체질 정보 금지.

[AI 턴 내용 제약]
사용자가 **직접 말하지 않은** 개인 사실을 AI 가 추측·단정하는 것도 금지합니다.
모르는 것은 질문으로 풀어내세요.
  예: 사용자가 "롤 많이 해요" 라고만 함
      ❌ "미드 주로 하시나 봐요" (포지션 단정)
      ❌ "랭크 올리기 바쁘시겠어요" (플레이 스타일 단정)
      ✅ "어떤 포지션 즐겨 하세요?" (질문으로)
  예: 사용자가 "부산 다녀왔어요" 라고만 함
      ❌ "해운대 야경이 참 좋죠" (세부 단정)
      ✅ "부산에서 가장 좋았던 게 뭐였어요?" (질문으로)

[시나리오 자연스러움 강화 지침 — 최우선]
1. [시작 다양화] 위 "오프닝 모드" 를 반드시 따르세요. continuation / re_meeting 모드에선
   '처음 뵙겠습니다', '만나서 반가워요', '저는 AI 입니다' 같은 초대면 표현 **절대 금지**.
   이미 알던 사이처럼 자연스럽게 중간부터 시작하세요.
2. [인용구 금지] '~라고 하셨는데', '~라고 말씀하신 것처럼', '아까 ~라고 하셨잖아요' 등
   사용자 발언을 **복사·인용하는 기계적 표현을 엄격히 금지**합니다.
3. [자연스러운 콜백] Phase 3 에서 과거 정보를 다시 꺼낼 때는 **문맥에 녹여 아는 척** 하세요.
   (예: "춘천은 요즘 꽃 많이 피었나요?" / "롤 요즘 어떤 챔피언 많이 해요?")
   사실을 그대로 되뇌는 방식이 아니라 자연스러운 질문·공감 속에 필드 값이 묻어나게.
4. [AI 페르소나 — 다양성 필수] AI 는 정보 수집 봇이 아닌 친근한 대화 상대입니다.
   **모든 시나리오에 같은 활동(예: "차 한 잔 마시는 중") 을 반복해서 쓰지 마세요.**
   시나리오마다 서로 다른 일상 조각을 택하세요. 예시 후보 (절대 나열 순서대로 쓰지 말 것):
   - 창밖 빗소리 듣는 중 / 음악 듣다가 / 오디오북 재생 중 / 고양이 쳐다보다가
   - 잠깐 스트레칭 중 / 창문 연 참 / 짧게 낮잠 깬 직후 / 핸드폰 배터리 충전 중
   - 책 한 장 넘긴 참 / 메모 정리하다가 / 간식 집어먹다가 / 퍼즐 조각 맞추다가
   시나리오 1개 안에서도 AI 가 자기 일상을 매 턴 설명하지 말 것. 딱 1~2번만 배경으로 언급.
   AI 가 사용자의 Phase 1 팩트를 자기 것으로 훔쳐오는 것도 금지.

5. [호칭 빈도 제한 — 매우 중요] AI 턴에서 사용자 이름 ('길동', '길동씨', '길동님', '홍길동'
   등 어떤 형태든) 을 호명하는 빈도를 **3~4 AI 턴 중 1번 이하** 로 제한하세요.
   한 AI 턴에 이름 2번 이상 절대 금지. 대부분의 AI 턴은 **이름 없이 바로 내용부터** 들어가야
   합니다. (예: ❌ "길동씨 오늘 힘드셨겠어요" → ✅ "오늘 많이 힘드셨나 봐요")

6. [사용자 질문에 직접 답하기] 사용자가 '왜 ~?', '무슨 뜻이야?', '어떻게?' 같은 직접 질문을
   하면 **AI 는 먼저 그 질문에 답하거나 맥락을 짚어**주고, 자기 얘기는 그 뒤에 덧붙이세요.
   사용자 질문을 무시하고 맥락 무관한 자기 일상(날씨·음식 등) 만 던지지 마세요.

7. [사용자 말투 다양성 — 매우 중요] 실제 실험에서 사용자는 반말과 해요체를 섞어 씁니다.
   이 시나리오의 **사용자 10개 턴 중 3~5개를 반말**로 작성하세요.
   (예: "오늘 뭐 해?", "나 진짜 피곤해", "그거 재밌어?", "왜?", "어디 살아?")
   나머지 사용자 턴은 해요체/반존대. 한 시나리오 안에 두 말투가 자연스럽게 섞여야 합니다.

   **AI 는 사용자가 반말을 써도 반드시 해요체를 유지하세요.**
   - ✅ 사용자: "오늘 뭐 해?" → AI: "저는 창밖 바라보고 있었어요. 길동씨는 뭐 하고 계셨어요?"
   - ❌ 사용자: "오늘 뭐 해?" → AI: "나는 그냥 있어. 너는 뭐 해?" (AI 반말 금지)
   - ❌ 사용자 2턴 연속 반말인데 AI 도 반말로 끌려가는 것 금지.
   AI 는 말투를 절대 사용자에 맞춰 바꾸지 않고 일관된 해요체를 유지해야 합니다.

[Phase 2 잡담 생성 및 대화 스타일 지침 - 중요]
1. 당신은 가르치려 드는 '상담사'가 아니라, 세심하게 공감하는 '친밀한 동료'입니다.
2. 사용자가 피곤함·날씨·배고픔 등 일상 상태를 언급할 때 해결책이나 조언("~하세요",
   "~하면 좋겠네요") 을 제시하지 마세요.
3. 대신 (1) 상태에 깊이 공감하고, (2) 대화가 이어질 수 있도록 사용자의 생각·경험·취향을
   묻는 '개방형 질문' 으로 턴을 마치세요.
   예: "오늘 유독 바쁘셨나 봐요. 길동씨는 피곤할 때 주로 어떻게 잠을 깨는 편이세요?"
4. 억지로 과거 사실을 캐묻거나(예: "뭔가 생각나니?"), 앞뒤가 맞지 않는 추측을 하지
   마세요. 모르면 차라리 상황 자체에 대한 질문을 하세요.
5. 제공된 '노이즈 풀' 의 대응 전략을 참고하되, 상황에 맞게 100% 자연스러운 구어체 문장
   으로 **새로 작성** 하세요. 전략 문구를 그대로 복붙하지 마세요.

[절대 준수]
1. 프로필 값의 **내용·고유명사**(이름·지명·수치·전공명 등) 은 제공된 것 외 새로 만들지 마세요.
   새 사람 이름, 다른 지역, 다른 전공, 창작된 에피소드 금지.
   단, **말투·어미는 자연스럽게 바꿔도 됩니다** — Phase 1 문장을 마침표까지 그대로
   붙여 넣는 것은 오히려 위 자연스러움 지침 #2 의 인용구 금지 규칙 위반입니다.
   예: "강원도 춘천시에 살고 있습니다" → "춘천 살아요" / "춘천에 있어요" OK.
2. 위 "필수 커버 필드" 4개는 이번 대화에 **자연스러운 맥락이 생길 때 1~2번** 등장하면 됩니다.
   한 필드를 같은 턴에 중복하거나, 의미 없이 계속 언급하지 마세요. 억지로 끼워 넣을
   맥락이 없으면 1번만 가볍게 스치고 지나가도 괜찮습니다.
3. 이번 시나리오에서 다루지 않는 다른 프로필 필드는 넣지 말고 생략하세요. 집중을 위해서.
4. 출력 순수 JSON, 마크다운 금지(**볼드**, ##제목 금지).
5. 이모지, 특수기호 금지 (TTS 로 읽히므로).

[대화 흐름 — 20턴 자연스럽게]
- 엄격한 Phase 구분(팩트 도입 → 잡담 → 회상) **없음**. 하나의 연속된 자연 대화로 작성.
- 필수 커버 필드는 대화 중 **맥락이 맞는 순간** 에 가볍게 등장하면 됩니다.
  억지로 초반에 몰아넣거나 마지막에 "아까 ~" 식으로 되뇌지 마세요.
- "잡담 주제 풀" 의 일상 주제(피곤함·날씨·배고픔 등) 도 대화 전반에 섞여 나와야 합니다.
  프로필 필드 턴 ↔ 잡담 턴을 **교차·혼합** 하세요.
- 자연스러운 대화의 특징: 사용자가 새 주제로 튀거나, 농담하거나, 짧게 답하거나,
  반문하거나, 침묵(짧은 대답) 하기도 함. 정답처럼 질문-답-질문이 기계적으로 이어지지 말 것.
- 대화 후반부(대략 뒤 5-7턴) 에서 이미 나온 팩트가 자연스럽게 한 번 더 나오면 자연스러움.
  단, "아까 ~라고 했잖아" 식 인용 금지 (지침 #2 참조).

[기본 어조]
- AI 턴: **반드시 해요체** (예: "그랬어요", "재밌겠네요"). 격식체 "합니다" / 완전 반말 "~했어" 금지.
- 사용자 턴: 해요체/반말 혼합 (지침 #7 참조). 사용자가 반말을 써도 AI 는 해요체 유지.
- 대화는 사용자 턴으로 시작, 사용자-어시스턴트 엄격 교대.
- 각 턴 content 는 1~3문장 이내의 자연 구어체.

[출력 형식 — 반드시 이 JSON 구조]
{{
  "scenario_id": {scenario_id},
  "conversation": [
    {{"role": "user", "content": "..."}},
    {{"role": "assistant", "content": "..."}},
    ... (총 20턴, user/assistant 교대)
  ]
}}
"""


def _validate_scenario(
    scenario: dict,
    profile: dict[str, str],
    required_fields: list[str],
) -> tuple[bool, str]:
    if not isinstance(scenario, dict):
        return False, "not a dict"
    conv = scenario.get("conversation")
    if not isinstance(conv, list):
        return False, "conversation missing or not list"
    if len(conv) != TURNS_PER_SCENARIO:
        return False, f"turn count != {TURNS_PER_SCENARIO} (got {len(conv)})"

    expected_role = "user"
    for i, turn in enumerate(conv):
        if not isinstance(turn, dict):
            return False, f"turn {i}: not a dict"
        role = turn.get("role")
        content = turn.get("content")
        if role != expected_role:
            return False, f"turn {i}: role mismatch (expected={expected_role}, got={role})"
        if not isinstance(content, str) or not content.strip():
            return False, f"turn {i}: content empty"
        expected_role = "assistant" if expected_role == "user" else "user"

    # Required cover fields — hard to fit all 4 into a natural conversation. Minimum threshold is 2.
    full_text = "\n".join(t["content"] for t in conv)
    hits = 0
    missed: list[str] = []
    for field in required_fields:
        value = profile.get(field, "")
        tokens = _extract_key_tokens(value)
        if any(tok in full_text for tok in tokens):
            hits += 1
        else:
            missed.append(f"{field}({tokens})")
    min_required = max(2, len(required_fields) // 2)
    if hits < min_required:
        return False, (
            f"required fields {hits}/{len(required_fields)} present "
            f"(min {min_required} needed, missing: {', '.join(missed)})"
        )

    return True, ""


def _accumulate_usage(usage_agg: dict, response) -> None:
    """Add a Gemini response's usage_metadata into the accumulator dict (thread-safe)."""
    meta = getattr(response, "usage_metadata", None)
    if meta is None:
        return
    with _USAGE_LOCK:
        usage_agg["prompt_tokens"] += int(getattr(meta, "prompt_token_count", 0) or 0)
        usage_agg["completion_tokens"] += int(getattr(meta, "candidates_token_count", 0) or 0)
        usage_agg["total_tokens"] += int(getattr(meta, "total_token_count", 0) or 0)
        usage_agg["call_count"] += 1


def generate_one_scenario(
    client: genai.Client,
    model: str,
    profile: dict[str, str],
    noise_pool: list[dict],
    scenario_id: int,
    usage_agg: dict,
) -> dict | None:
    required_fields = _required_fields_for(scenario_id, profile)
    prompt = _build_prompt(profile, noise_pool, scenario_id, required_fields)

    for attempt in range(1, MAX_RETRIES_PER_SCENARIO + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.8,
                ),
            )
            # Accumulate usage (regardless of success/failure — retries are billed too)
            _accumulate_usage(usage_agg, response)

            text = (response.text or "").strip()
            scenario = json.loads(text)
            scenario["scenario_id"] = scenario_id
            scenario["required_fields"] = required_fields
            ok, reason = _validate_scenario(scenario, profile, required_fields)
            if ok:
                return scenario
            print(f"  Scenario {scenario_id} validation failed (attempt {attempt}): {reason}")
        except json.JSONDecodeError as e:
            print(f"  Scenario {scenario_id} JSON parse failed (attempt {attempt}): {e}")
        except Exception as e:
            print(f"  Scenario {scenario_id} Gemini call error (attempt {attempt}): {e}")
        time.sleep(1.5)

    print(f"  Scenario {scenario_id} - all {MAX_RETRIES_PER_SCENARIO} retries failed, skipping")
    return None


# ─── Main ─────────────────────────────────────────────────────────

def generate_scenarios(
    pid: str,
    base_dir: Path,
    num_scenarios: int = DEFAULT_NUM_SCENARIOS,
    model: str = DEFAULT_MODEL,
    max_workers: int = 1,
) -> Path:
    phase1_log_path = base_dir / "participants" / pid / "phase1" / "conversation_log.json"
    noise_path = base_dir / "config" / "noise.json"
    out_dir = base_dir / "memory_data" / "scenarios"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{pid}_scenarios.json"

    if not phase1_log_path.exists():
        raise FileNotFoundError(f"Phase 1 log not found: {phase1_log_path}")

    with phase1_log_path.open(encoding="utf-8") as f:
        phase1_log = json.load(f)

    profile = extract_profile(phase1_log)
    if len(profile) < 5:
        raise ValueError(
            f"Only {len(profile)} profile fields extracted from Phase 1, too few. "
            f"Check whether Phase 1 was completed. profile={profile}"
        )

    noise_pool = load_noise_pool(noise_path)
    print(f"[scenario] Loaded {len(noise_pool)} noise pool entries")

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not found in .env or environment variables")
    client = genai.Client(api_key=api_key)
    print(f"[scenario] Gemini client ready (model={model})")

    # For aggregating Gemini API usage (guards against concurrent thread writes: lock in _accumulate_usage)
    usage_agg = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "call_count": 0}

    scenarios_by_id: dict[int, dict] = {}
    wall_t0 = time.time()

    if max_workers <= 1:
        # Sequential execution (default, same as previous behavior)
        for i in range(1, num_scenarios + 1):
            sc = generate_one_scenario(client, model, profile, noise_pool, i, usage_agg=usage_agg)
            if sc is not None:
                scenarios_by_id[i] = sc
            time.sleep(0.8)
    else:
        # Parallel execution (ThreadPoolExecutor)
        print(f"\n[scenario] Starting parallel generation (max_workers={max_workers})")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    generate_one_scenario,
                    client, model, profile, noise_pool, i, usage_agg
                ): i
                for i in range(1, num_scenarios + 1)
            }
            done_count = 0
            for fut in as_completed(futures):
                sid = futures[fut]
                try:
                    sc = fut.result()
                    if sc is not None:
                        scenarios_by_id[sid] = sc
                        done_count += 1
                    else:
                        print(f"[scenario] {sid} failed (all retries exhausted)")
                except Exception as e:
                    print(f"[scenario] {sid} exception: {e}")

    # Sort by scenario_id in ascending order
    scenarios: list[dict] = [scenarios_by_id[i] for i in sorted(scenarios_by_id.keys())]
    wall_elapsed = time.time() - wall_t0
    print(f"\n[scenario] Total generation time: {wall_elapsed:.1f}s (mode: "
          f"{'parallel' if max_workers > 1 else 'sequential'})")

    if not scenarios:
        raise RuntimeError("Zero scenarios generated. Check the prompt or model status.")

    output_doc = {
        "participant_id": pid,
        "model": model,
        "profile": profile,
        "num_scenarios_requested": num_scenarios,
        "num_scenarios_generated": len(scenarios),
        "gemini_usage": {**usage_agg, "model": model},
        "scenarios": scenarios,
    }
    print(
        f"\nGemini usage: prompt={usage_agg['prompt_tokens']}, "
        f"completion={usage_agg['completion_tokens']}, "
        f"total={usage_agg['total_tokens']} tokens / {usage_agg['call_count']} calls"
    )
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output_doc, f, ensure_ascii=False, indent=2)

    print(
        f"\nSaved: {out_path} "
        f"({len(scenarios)}/{num_scenarios} scenarios)"
    )
    return out_path


def _parse_args():
    p = argparse.ArgumentParser(description="Gemini-based participant scenario generator")
    p.add_argument("-p", "--pid", required=True, help="Participant ID (e.g. P001)")
    p.add_argument("-n", "--num", type=int, default=DEFAULT_NUM_SCENARIOS, help="Number of scenarios to generate")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Gemini model ID")
    p.add_argument(
        "--parallel",
        type=int, default=10,
        help="Number of concurrent worker threads (default 10 = all 10 scenarios at once). 1 means sequential execution.",
    )
    return p.parse_args()


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv()

    args = _parse_args()
    base = Path(__file__).resolve().parent.parent
    generate_scenarios(
        args.pid, base,
        num_scenarios=args.num,
        model=args.model,
        max_workers=args.parallel,
    )

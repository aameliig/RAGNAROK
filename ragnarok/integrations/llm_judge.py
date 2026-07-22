import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional


class JudgeCriterion(Enum):
    FAITHFULNESS = "faithfulness"
    RELEVANCE = "relevance"
    COMPLETENESS = "completeness"
    COHERENCE = "coherence"
    CONCISENESS = "conciseness"


@dataclass
class JudgeResult:
    criterion: JudgeCriterion
    score: float
    explanation: str
    confidence: float


class BaseLLM(ABC):
    @abstractmethod
    def complete(self, prompt: str, temperature: float = 0.0) -> str:
        pass

    @abstractmethod
    def name(self) -> str:
        pass


class OpenAIJudge(BaseLLM):
    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("pip install openai")

        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("Set OPENAI_API_KEY env var")

        self.client = OpenAI(api_key=self.api_key)
        self.model = model

    def complete(self, prompt: str, temperature: float = 0.0) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=500,
        )
        return response.choices[0].message.content

    def name(self) -> str:
        return f"openai:{self.model}"


class GigaChatJudge(BaseLLM):
    def __init__(self, model: str = "GigaChat",
                 client_id: Optional[str] = None,
                 client_secret: Optional[str] = None):
        self.model = model
        self.client_id = client_id or os.environ.get("GIGACHAT_CLIENT_ID")
        self.client_secret = client_secret or os.environ.get("GIGACHAT_CLIENT_SECRET")

        if not self.client_id or not self.client_secret:
            raise ValueError("Set GIGACHAT_CLIENT_ID and GIGACHAT_CLIENT_SECRET env vars")

    def _get_token(self) -> str:
        import requests
        import uuid
        url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
        headers = {
            "Authorization": f"Basic {self.client_id}:{self.client_secret}",
            "RqUID": str(uuid.uuid4()),
            "Content-Type": "application/x-www-form-urlencoded",
        }
        data = {"scope": "GIGACHAT_API_PERS"}
        response = requests.post(url, headers=headers, data=data, verify=False)
        response.raise_for_status()
        return response.json()["access_token"]

    def complete(self, prompt: str, temperature: float = 0.0) -> str:
        import requests
        token = self._get_token()
        url = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": 500,
        }
        response = requests.post(url, headers=headers, json=body, verify=False)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def name(self) -> str:
        return f"gigachat:{self.model}"


class LLMJudge:
    RUBRICS = {
        JudgeCriterion.FAITHFULNESS: """Оцени faithfulness ответа относительно контекста.
1 — полные галлюцинации, 3 — частично верно, 5 — идеально.
Сначала объясни, потом Score: X""",
        JudgeCriterion.RELEVANCE: """Оцени релевантность ответа вопросу.
1 — не по теме, 3 — частично, 5 — полностью отвечает.
Сначала объясни, потом Score: X""",
        JudgeCriterion.COMPLETENESS: """Оцени полноту ответа.
1 — пропущено всё важное, 3 — частично, 5 — всё учтено.
Сначала объясни, потом Score: X""",
        JudgeCriterion.COHERENCE: """Оцени связность и логичность.
1 — бессвязный, 3 — терпимо, 5 — идеально структурирован.
Сначала объясни, потом Score: X""",
        JudgeCriterion.CONCISENESS: """Оцени лаконичность.
1 — много воды, 3 — есть избыточность, 5 — только суть.
Сначала объясни, потом Score: X""",
    }

    def __init__(self, llm: BaseLLM,
                 criteria: Optional[List[JudgeCriterion]] = None):
        self.llm = llm
        self.criteria = criteria or [JudgeCriterion.FAITHFULNESS, JudgeCriterion.RELEVANCE]

    def evaluate(self, query: str, answer: str, contexts: List[str]) -> Dict[str, JudgeResult]:
        context_text = "\n".join(contexts)
        results = {}

        for criterion in self.criteria:
            rubric = self.RUBRICS.get(criterion, "")
            prompt = f"""[Instruction]
Ты — эксперт-оценщик RAG-систем. Оцени качество ответа объективно.

[Context]
{context_text}

[Question]
{query}

[Answer]
{answer}

[Criteria]
{rubric}

[Format]
Reasoning: <объяснение>
Score: <число 1-5>"""

            try:
                response = self.llm.complete(prompt, temperature=0.0)
                score, explanation = self._parse(response)
                results[criterion.value] = JudgeResult(
                    criterion=criterion,
                    score=score / 5.0,
                    explanation=explanation,
                    confidence=self._confidence(response, score),
                )
            except Exception as e:
                results[criterion.value] = JudgeResult(
                    criterion=criterion, score=0.0,
                    explanation=f"Error: {str(e)}", confidence=0.0,
                )

        return results

    def _parse(self, response: str) -> tuple:
        lines = response.strip().split('\n')
        for line in reversed(lines):
            if "score" in line.lower():
                match = re.search(r'(\d+(?:\.\d+)?)', line)
                if match:
                    return max(1.0, min(5.0, float(match.group(1)))), response
        return 3.0, response

    def _confidence(self, response: str, score: float) -> float:
        has_reasoning = len(response) > 50
        has_score = "score:" in response.lower()
        conf = 0.5
        if has_reasoning: conf += 0.3
        if has_score: conf += 0.2
        return min(1.0, conf)

    @classmethod
    def create(cls, provider: str, model: str, **kwargs) -> "LLMJudge":
        if provider == "openai":
            llm = OpenAIJudge(model=model, **kwargs)
        elif provider == "gigachat":
            llm = GigaChatJudge(model=model, **kwargs)
        else:
            raise ValueError(f"Unknown: {provider}. Use: openai, gigachat")
        return cls(llm=llm)

    @classmethod
    def from_env(cls) -> "LLMJudge":
        if os.environ.get("OPENAI_API_KEY"):
            return cls.create("openai", os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
        elif os.environ.get("GIGACHAT_CLIENT_ID"):
            return cls.create("gigachat", "GigaChat")
        raise ValueError("No API keys found. Set OPENAI_API_KEY or GIGACHAT_CLIENT_ID")

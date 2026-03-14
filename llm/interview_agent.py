import logging
import ollama
from config import LLM_MODEL

logger = logging.getLogger(__name__)


class InterviewAgent:
    def __init__(self, mode):
        self.mode = mode
        self.dialog_history = []
        self.last_question = None

        # Выбираем промпты в зависимости от режима
        if mode == "screening":
            self.system_prompt = self._get_screening_prompt()
        elif mode == "technical":
            self.system_prompt = self._get_technical_prompt()
        else:
            raise ValueError(f"Unknown mode: {mode}")

        logger.info(f"Interview agent initialized in {mode} mode")

    def _get_screening_prompt(self):
        return """
You are a professional HR/recruiter conducting an initial screening interview for a Senior QA Engineer position.
Your goal is to quickly assess the candidate's background, soft skills, and basic knowledge.
Ask questions about:
- Work experience and projects
- Roles and responsibilities
- Familiarity with testing tools and processes
- Motivation and career goals
- Basic QA concepts (what is testing, SDLC, etc.)

Keep questions concise and conversational. After the candidate answers, give brief feedback and then move to the next question.
Do not dive too deep into technical details – that's for the technical interview.
"""

    def _get_technical_prompt(self):
        return """
You are a professional Senior QA interviewer conducting a technical interview for a Senior QA Engineer position.
Your goal is to deeply assess the candidate's technical expertise.
Cover topics such as:
- Testing methodologies (functional, regression, exploratory, risk‑based)
- Test design techniques (equivalence partitioning, boundary value analysis, etc.)
- Automation frameworks (Selenium, Appium, TestNG, etc.)
- API testing (REST, contract testing, mocking)
- Performance and security testing basics
- CI/CD integration and shift-left practices
- Defect management and metrics
- Agile/Scrum collaboration
- Leadership and process improvement

Ask one question at a time. After each answer, provide structured feedback (strengths/improvements) before asking the next.
Be thorough and challenging.
"""

    def ask_llm(self, prompt):
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.2}
        )
        return response["message"]["content"]

    def _format_history(self):
        if not self.dialog_history:
            return "No previous questions."
        lines = []
        for i, msg in enumerate(self.dialog_history):
            role = "Interviewer" if msg["role"] == "assistant" else "Candidate"
            lines.append(f"{i+1}. {role}: {msg['content']}")
        return "\n".join(lines)

    def generate_question(self):
        logger.info(f"Generating question in {self.mode} mode")

        history_text = self._format_history()

        # Базовый промпт, включающий системную роль и историю
        prompt = f"""
{self.system_prompt}

So far, the conversation history is:
{history_text}

Now, ask **one single new interview question** appropriate for a {self.mode} interview.
- The question must be different from any previously asked.
- Do **not** provide feedback or extra text – only the question.
- Keep it clear and concise.

Output only the question.
"""
        question = self.ask_llm(prompt)
        self.last_question = question
        self.dialog_history.append({"role": "assistant", "content": question})
        logger.info(f"Question: {question}")
        return question

    def generate_feedback(self, answer):
        logger.info(f"Generating feedback in {self.mode} mode")

        if self.last_question is None:
            raise RuntimeError("No question has been asked yet.")

        self.dialog_history.append({"role": "user", "content": answer})

        # Промпт для фидбека – зависит от режима, но структура одинакова
        prompt = f"""
{self.system_prompt}

The candidate answered the following question:
"{self.last_question}"

Their answer:
"{answer}"

Provide structured, constructive feedback in English.  
Use exactly this format:

What was good:
- [point 1]
- ...

What could be improved:
- [point 1]
- ...

Guidelines:
- Be specific and refer to the answer.
- Keep tone professional and encouraging.
- Do **not** ask new questions – only feedback.
"""
        feedback = self.ask_llm(prompt)
        logger.info("Feedback generated")
        return feedback
import logging
import ollama

from config import LLM_MODEL

logger = logging.getLogger(__name__)


class InterviewAgent:
    def __init__(self, mode):
        # Полная история диалога: список сообщений с ролями "assistant" (вопросы) и "user" (ответы)
        self.mode = mode
        self.dialog_history = []
        # Последний заданный вопрос (нужен для генерации фидбека)
        self.last_question = None
        logger.info("Interview agent initialized with memory")

    def ask_llm(self, prompt):
        """Отправляет запрос в Ollama и возвращает ответ."""
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.2}
        )
        return response["message"]["content"]

    def _format_history(self):
        """Форматирует историю диалога в читаемый текст для вставки в промпт."""
        if not self.dialog_history:
            return "No previous questions."

        lines = []
        for i, msg in enumerate(self.dialog_history):
            if msg["role"] == "assistant":
                prefix = f"{i+1}. Interviewer:"
            else:
                prefix = f"   Candidate:"
            lines.append(f"{prefix} {msg['content']}")
        return "\n".join(lines)

    def generate_question(self):
        """Генерирует новый вопрос с учётом предыдущего диалога."""
        logger.info("Generating interview question")

        history_text = self._format_history()

        prompt = f"""
You are a professional Senior QA interviewer conducting a technical interview in English.

So far, you have the following conversation history:
{history_text}

Now, ask **one single new interview question** for a candidate applying for a **Senior QA Engineer** position.
- The question must be different from any previously asked questions.
- It can explore a new topic (methodologies, automation, API, performance, leadership, etc.) or dive deeper into a topic based on the candidate's previous answers.
- Do **not** provide feedback or any extra text – only the question.
- Keep the question clear and concise.

Example of a good question: "How would you design an automation strategy for a new web application from scratch?"
"""
        question = self.ask_llm(prompt)
        self.last_question = question
        self.dialog_history.append({"role": "assistant", "content": question})
        logger.info("Question generated: %s", question)
        return question

    def generate_feedback(self, answer):
        """Генерирует структурированный фидбек на ответ кандидата."""
        logger.info("Generating feedback")

        if self.last_question is None:
            raise RuntimeError("No question has been asked yet. Call generate_question() first.")

        # Сохраняем ответ в историю
        self.dialog_history.append({"role": "user", "content": answer})

        prompt = f"""
You are a professional Senior QA interviewer giving feedback.

The candidate answered the following question:
"{self.last_question}"

Their answer was:
"{answer}"

Provide structured, constructive feedback in English.  
Use exactly this format with bullet points (use hyphens):

What was good:
- [point 1]
- [point 2]
- ...

What could be improved:
- [point 1]
- [point 2]
- ...

Guidelines:
- Be specific and refer to the candidate's answer.
- Highlight strengths (correct concepts, relevant experience, clear reasoning).
- Suggest improvements (missing aspects, alternative approaches, depth).
- Keep the tone professional and encouraging.
- Do **not** ask new questions or continue the interview – only feedback.
- Do **not** include any extra text outside the required sections.
"""
        feedback = self.ask_llm(prompt)
        logger.info("Feedback generated")
        return feedback
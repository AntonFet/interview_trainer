import logging
import ollama

from config import LLM_MODEL

logger = logging.getLogger(__name__)


class InterviewAgent:

    def ask_llm(self, prompt):

        response = ollama.chat(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": 0.2
            }
        )

        return response["message"]["content"]

    def generate_question(self):

        logger.info("Generating interview question")

        prompt = """
You are a professional Senior QA interviewer conducting a technical interview in English.

Generate **one single interview question** for a candidate applying for a **Senior QA Engineer** position.  
The question must be relevant to the senior level and can cover any of the following areas (choose one per question, vary topics over time):

- Testing methodologies (functional, regression, exploratory, risk‑based)
- Test design techniques (equivalence partitioning, boundary value analysis, state transition, etc.)
- Automation strategy, framework design, and maintenance (Selenium, Appium, TestNG, etc.)
- API testing (REST, contract testing, mocking, tools like Postman)
- Performance testing (types, tools, metrics, how to approach it)
- Security testing (basic concepts, common vulnerabilities, shift‑left)
- CI/CD integration and test automation pipelines
- Defect management, bug reporting, and metrics
- Agile/Scrum practices and team collaboration
- Leadership, mentoring, and process improvement
- Handling ambiguous requirements and trade‑offs

Output **only the question** – no greetings, no explanations, no extra text.

Example of a good question:  
"How would you design an automation strategy for a new web application from scratch?"
"""

        return self.ask_llm(prompt)

    def generate_feedback(self, answer):

        logger.info("Generating feedback")

        prompt = f"""
You are a professional Senior QA interviewer giving feedback to a candidate after an interview question.

Candidate's answer:
{answer}

Provide structured, constructive feedback in English.  
Use exactly the following format, with bullet points (use hyphens):

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

        return self.ask_llm(prompt)
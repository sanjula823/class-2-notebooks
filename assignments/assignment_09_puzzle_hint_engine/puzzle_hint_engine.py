"""
Assignment 9: Puzzle Hint Engine (Difficulty Controls)

Goal: Generate layered hints for a simple riddle or logic puzzle, adapting
verbosity and directness by `difficulty`.
"""

import os
from typing import List
from pydantic import BaseModel, Field

from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser



class Hint(BaseModel):
    """Structured hint output."""

    level: int = Field(..., description="1=light nudge, higher=more direct")
    text: str

class HintsWrapper(BaseModel):
    hints: List[Hint]


class PuzzleHintEngine:
    """Produce hints without giving away the answer at low difficulty.

    Use structured outputs or JSON parsing for consistency.
    At higher difficulty values, hints should be vaguer; at lower values, more direct.
    """

    def __init__(self):
        """Prepare prompt strings and placeholders.

        Provide:
        - `system_prompt` describing progressive hinting philosophy.
        - `user_prompt` with variables {attempt}, {difficulty}, {puzzle}.
        - A structured-output LLM placeholder (None) and TODO to create it.
        """
        self.system_prompt = "You provide puzzle hints in progressive layers, never spoiling unless difficulty is very low."
        self.user_prompt = (
              "Puzzle: {puzzle}\nAttempt: {attempt}\nDifficulty: {difficulty}\n"
              "Return a JSON object with key 'hints' containing 2-3 hints, each with 'level' and 'text'."
         )

        # TODO: Build prompt and a structured-output LLM targeting List[Hint]
        

        # Build the prompt template
        self.prompt = ChatPromptTemplate.from_template(self.user_prompt)

        # Build structured-output parser
        self.parser = PydanticOutputParser(model=HintsWrapper)

        # Build the LLM
        self.llm = ChatOpenAI(temperature=0.5, model_name="gpt-4o-mini")

        # Combine into a chain
        self.chain = LLMChain(prompt=self.prompt, llm=self.llm, output_parser=self.parser)


    def get_hints(self, puzzle: str, attempt: str, difficulty: int = 3) -> List[Hint]:
        """Return 2-3 hints tailored to the attempt and difficulty.

        Implement:
        - Wire prompt→llm→structured parser (e.g., with Pydantic) and invoke.
        - Ensure output is parsed into a list of `Hint` models.
        """
        inputs = {
          "puzzle": puzzle,
          "attempt": attempt,
          "difficulty": difficulty
          }

        result = self.chain.invoke(inputs)
        return result.hints  


def _demo():
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ Set OPENAI_API_KEY before running.")
    engine = PuzzleHintEngine()
    try:
        print("\n🧩 Puzzle Hint Engine — demo\n" + "-" * 40)
        hints = engine.get_hints(
            "I speak without a mouth and hear without ears.",
            attempt="Is it wind?",
            difficulty=2,
        )
        for h in hints:
            print(f"[{h.level}] {h.text}")
    except NotImplementedError as e:
        print(e)


if __name__ == "__main__":
    _demo()

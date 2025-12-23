"""
Assignment 8: Micro-Coach (On-Demand Streaming)

Goal: Provide a short plan non-streamed, and when `stream=True` deliver
encouraging guidance token-by-token via a callback.
"""

import os
from typing import Any
from langchain.callbacks.base import BaseCallbackHandler
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class PrintTokens(BaseCallbackHandler):
    """Print tokens as they arrive (streaming)."""
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(token, end="")



class MicroCoach:
    def __init__(self):
        """Store prompt strings and prepare placeholders.

        Provide:
        - `system_prompt` motivating but practical tone
        - `user_prompt` with variables {goal}, {time_available}
        - `self.llm_streaming` and `self.llm_plain` placeholders (None), with TODOs
        - `self.stream_prompt` and `self.plain_prompt` placeholders (None), with TODOs
        """
        self.system_prompt = (
            "You are a supportive micro-coach. Keep plans realistic and brief."
        )
        self.user_prompt = "Goal: {goal}\nTime: {time_available}\nReturn a 3-step plan."

        # TODO: Build prompts and LLMs (streaming and non-streaming)
        # Build ChatPromptTemplates
        self.stream_prompt = ChatPromptTemplate.from_messages([
          ("system", self.system_prompt),
          ("user", self.user_prompt)
        ])
        self.plain_prompt = ChatPromptTemplate.from_messages([
          ("system", self.system_prompt),
          ("user", self.user_prompt)
           ])   

       # Create streaming LLM with callback
        self.llm_streaming = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.5,
        streaming=True,
        callbacks=[PrintTokens()]
        )

       # Create non-streaming LLM
        self.llm_plain = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3
       )

       # Build chains
        self.stream_chain = self.stream_prompt | self.llm_streaming | StrOutputParser()
        self.plain_chain = self.plain_prompt | self.llm_plain | StrOutputParser()


    def coach(self, goal: str, time_available: str, stream: bool = False) -> str:
        if stream:
        # Stream output token by token
         _ = self.stream_chain.invoke({"goal": goal, "time_available": time_available})
         print()  # newline after streaming
         return ""
        else:
        # Non-streamed compact plan
         return self.plain_chain.invoke({"goal": goal, "time_available": time_available})


def _demo():
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ Set OPENAI_API_KEY before running.")
    coach = MicroCoach()
    try:
        print("\n🏃 Micro-Coach — demo\n" + "-" * 40)
        print(coach.coach("resume drafting", "25 minutes", stream=False))
        print()
        print("\nStreaming example:")
        coach.coach("push-ups habit", "10 minutes", stream=True)
        print()
    except NotImplementedError as e:
        print(e)


if __name__ == "__main__":
    _demo()

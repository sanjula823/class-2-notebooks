"""
Assignment 7: Minutes & Action Items Batcher

Goal: Convert meeting transcripts into concise minutes and action items, with
support for batch processing many transcripts at once.
"""

import os
from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


class MinutesBatcher:
    """Summarize transcripts into minutes and action items.

    Implementations should use a prompt → llm → parser chain and demonstrate
    `.batch()` for parallel processing.
    """

    def __init__(self):
        """Prepare prompt strings and placeholders for the chain.

        Provide:
        - `system_prompt`: clear structure for minutes and actions.
        - `user_prompt`: variables {transcript}, {title}.
        - Do not build templates or chains here; keep them None with TODOs.
        """
        self.system_prompt = "You produce crisp meeting minutes and bullet action items with owners and due dates."
        self.user_prompt = (
            "Title: {title}\nTranscript:\n{transcript}\n\n"
            "Return sections: MINUTES (3-5 bullets), ACTIONS (bullets with owner;date)."
        )
        # TODO: Build ChatPromptTemplate and store as self.prompt
        self.prompt = ChatPromptTemplate.from_messages([
             ("system", self.system_prompt),
             ("user", self.user_prompt)
       ])
        # TODO: Create a low-temperature ChatOpenAI and store as self.llm
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

        # TODO: Build a chain `self.chain` with StrOutputParser
        self.chain = self.prompt | self.llm | StrOutputParser()

    def summarize_one(self, title: str, transcript: str) -> str:
        """Return minutes+actions for a single transcript.

        Implement using the prepared chain and `{title, transcript}` inputs.
        """
        return self.chain.invoke({"title": title, "transcript": transcript})

    def summarize_batch(self, items: List[Dict[str, str]]) -> List[str]:
        """Return minutes+actions for a batch of transcripts.

        Implement: use `.batch()` on the chain with a list of input dicts.
        Preserve order of inputs in the returned results.
        """
        return self.chain.batch([
             {"title": item["title"], "transcript": item["transcript"]}
             for item in items
         ])
 

def _demo():
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ Set OPENAI_API_KEY before running.")
    mb = MinutesBatcher()
    try:
        print("\n📝 Minutes & Actions — demo\n" + "-" * 40)
        print(
            mb.summarize_one(
                "Sprint Planning",
                "Discussed backlog grooming, two blockers, and deployment window next Tuesday.",
            )
        )
    except NotImplementedError as e:
        print(e)


if __name__ == "__main__":
    _demo()

from src.InfoGather.info_book import InfoBook
from src.LLM import AgentTool


class InfoBookTool(AgentTool):
    def __init__(self, info_book: InfoBook):
        self.info_book = info_book


__all__ = ["AgentTool", "InfoBookTool"]

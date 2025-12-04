#app/agent_builder.py

import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.output_parsers import StrOutputParser

from app.tools import RetrieverTool, calc_tool, summarizer_tool , WebSearchTool

# --------------------------------------------------
# GROQ LLM
# --------------------------------------------------
def build_groq_llm():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set")

    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.0,
        api_key=api_key
    )

# --------------------------------------------------
# Build Tools (LCEL)
# --------------------------------------------------
def build_tools(retriever):

    retr = RetrieverTool(retriever)
    tavily = WebSearchTool(os.environ.get("TAVILY_API_KEY"))

    @tool
    def search_docs(query: str) -> str:
        """Retrieve relevant medical documents."""
        docs = retr.run(query)
        return "\n\n".join([d.page_content for d in docs])
    
    @tool
    def web_search(query: str) -> str:
        """Search the real-time web for fresh info."""
        return tavily.run(query)

    @tool
    def calculator(expr: str) -> str:
        """Evaluate math expressions."""
        return calc_tool(expr)

    @tool
    def summarizer(text: str) -> str:
        """Summarize long text."""
        return summarizer_tool([text])

    return [search_docs, calculator, summarizer, web_search]

# --------------------------------------------------
# Very Simple Agent using LCEL
# --------------------------------------------------
class SimpleAgent:

    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = {t.name: t for t in tools}

        self.prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an Agentic RAG system.\n"
        "If you need outside info, respond ONLY with:\n"
        "{{ \"action\": \"tool_name\", \"input\": \"value\" }}\n"
        "Available tools: search_docs, web_search, calculator, summarizer.\n"
        "Prefer search_docs for local knowledge and web_search for real-time info.\n"
    ),
    ("human", "{input}")
])


        self.parser = StrOutputParser()

    def invoke(self, inp: dict):

        user_text = inp["input"]
        msgs = self.prompt.format_messages(input=user_text)

        response = self.llm.invoke(msgs)
        text = response.content.strip()

        # ------------------------------
        # 1) TRY STRICT JSON PARSE
        # ------------------------------
        def try_parse_json(s):
            import json, re

            # replace smart quotes
            s = s.replace("“", "\"").replace("”", "\"").replace("’", "'")

            # extract JSON object from anywhere in text
            match = re.search(r"\{.*\}", s, flags=re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except:
                    return None
            return None

        data = try_parse_json(text)

        if data and "action" in data:
            tool_name = data["action"]
            tool_input = data.get("input", "")

            tool = self.tools.get(tool_name)
            if not tool:
                return {"output": f"Unknown tool: {tool_name}"}

            tool_result = tool.run(tool_input)

            # ask LLM for final answer
            follow = self.llm.invoke([
                ("human", f"Tool result:\n{tool_result}\nANSWER USER DIRECTLY.")
            ])
            return {"output": follow.content}

        # ------------------------------
        # No JSON tool call → normal reply
        # ------------------------------
        return {"output": text}


# --------------------------------------------------
# Build Agent
# --------------------------------------------------
def build_agent(llm, retriever):
    tools = build_tools(retriever)
    return SimpleAgent(llm, tools)

# # app/agent_builder.py

# import os
# from langchain_groq import ChatGroq
# from langchain_core.tools import tool
# from langchain_core.prompts import ChatPromptTemplate

# from app.tools import RetrieverTool, calc_tool, summarizer_tool


# # --------------------------------------------------
# # Build GROQ LLM
# # --------------------------------------------------
# def build_groq_llm():
#     api_key = os.environ.get("GROQ_API_KEY")
#     if not api_key:
#         raise RuntimeError("GROQ_API_KEY not set in .env")

#     return ChatGroq(
#         model="mixtral-8x7b",
#         temperature=0.0,
#         api_key=api_key
#     )


# # --------------------------------------------------
# # Build LCEL Tools (new LangChain style)
# # --------------------------------------------------
# def build_tools(retriever):

#     retr_tool = RetrieverTool(retriever)

#     @tool
#     def retriever_tool(query: str) -> str:
#         """Retrieve relevant documents using Pinecone retriever."""
#         docs = retr_tool.run(query)
#         return "\n\n".join([d.page_content for d in docs])

#     @tool
#     def calculator(expr: str) -> str:
#         """Evaluate math expressions."""
#         return calc_tool(expr)

#     @tool
#     def summarizer(text: str) -> str:
#         """Summarize long text."""
#         return summarizer_tool([text])

#     return [retriever_tool, calculator, summarizer]


# # --------------------------------------------------
# # Very Simple Agent (LCEL-based)
# # --------------------------------------------------
# class SimpleAgent:

#     def __init__(self, llm, tools):
#         self.llm = llm
#         self.tools = {t.name: t for t in tools}

#         self.prompt = ChatPromptTemplate.from_messages([
#             (
#                 "system",
#                 "You are an Agentic RAG system.\n"
#                 "If you need outside info, respond ONLY with:\n"
#                 "{ \"action\": \"tool_name\", \"input\": \"value\" }\n"
#                 "Else answer normally.\n"
#             ),
#             ("human", "{input}")
#         ])

#     def invoke(self, query: str):

#         # Ask LLM what action to take
#         msgs = self.prompt.format_messages(input=query)
#         response = self.llm.invoke(msgs)
#         text = response.content.strip()

#         # Detect JSON tool call
#         if text.startswith("{") and "\"action\"" in text:
#             import json
#             data = json.loads(text)

#             tool_name = data["action"]
#             tool_input = data["input"]

#             tool = self.tools.get(tool_name)
#             if not tool:
#                 return {"output": f"Unknown tool: {tool_name}"}

#             tool_output = tool.run(tool_input)

#             # LLM converts tool result → final answer
#             follow = self.llm.invoke([
#                 ("human", f"Tool result:\n{tool_output}\nProvide final answer.")
#             ])

#             return {"output": follow.content}

#         # No tool needed
#         return {"output": text}


# # --------------------------------------------------
# # Build Agent
# # --------------------------------------------------
# def build_agent(llm, retriever):
#     tools = build_tools(retriever)
#     return SimpleAgent(llm, tools)





import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.output_parsers import StrOutputParser

from app.tools import RetrieverTool, calc_tool, summarizer_tool

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

    @tool
    def search_docs(query: str) -> str:
        """Retrieve relevant medical documents."""
        docs = retr.run(query)
        return "\n\n".join([d.page_content for d in docs])

    @tool
    def calculator(expr: str) -> str:
        """Evaluate math expressions."""
        return calc_tool(expr)

    @tool
    def summarizer(text: str) -> str:
        """Summarize long text."""
        return summarizer_tool([text])

    return [search_docs, calculator, summarizer]

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
        "Else answer normally.\n"
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

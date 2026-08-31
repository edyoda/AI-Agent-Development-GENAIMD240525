from typing import List, TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, StateGraph

from src.upstream.nodes import add_to_vectordb, chunk_docs, parse_docs


class GraphState(TypedDict):
    doc_path: str
    doc_name: str
    raw_text: str
    chunks: List[Document]
    message: str


def build_upstream_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("parse_docs", parse_docs)
    workflow.add_node("chunk_docs", chunk_docs)
    workflow.add_node("add_to_vectordb", add_to_vectordb)

    workflow.set_entry_point("parse_docs")
    workflow.add_edge("parse_docs", "chunk_docs")
    workflow.add_edge("chunk_docs", "add_to_vectordb")
    workflow.add_edge("add_to_vectordb", END)

    return workflow.compile()


def run_upstream(doc_path: str):
    graph = build_upstream_graph()
    result = graph.invoke({
        "doc_path": doc_path,
        "doc_name": "",
        "raw_text": "",
        "chunks": [],
        "message": "",
    })
    print(result["message"])
    return result

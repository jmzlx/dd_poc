"""LangGraph + Anthropic Integration for DD-Checklist"""
import os
import streamlit as st
from typing import Optional, List, Dict, TypedDict, Annotated, Sequence, Any
from typing_extensions import TypedDict
import json
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor

try:
    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
    from langgraph.graph import StateGraph, END
    from langgraph.prebuilt import ToolNode  # Updated from ToolExecutor
    from langchain_core.tools import tool
    from langgraph.checkpoint.memory import MemorySaver  # Updated import path
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False


# Define the state for our agent
class AgentState(TypedDict):
    """State for the due diligence agent"""
    messages: Sequence[BaseMessage]
    checklist: Optional[Dict]
    documents: Optional[List[Dict]]
    current_task: Optional[str]
    findings: Dict[str, List[str]]
    next_action: Optional[str]


class TaskType(Enum):
    """Types of tasks the agent can perform"""
    PARSE_CHECKLIST = "parse_checklist"
    ANALYZE_DOCUMENT = "analyze_document"
    MATCH_CHECKLIST = "match_checklist"
    ANSWER_QUESTION = "answer_question"
    SUMMARIZE_FINDINGS = "summarize_findings"


def get_langgraph_agent(api_key: Optional[str] = None, model: str = "claude-3-haiku-20240307"):
    """Create a LangGraph agent with Anthropic"""
    
    if not LANGGRAPH_AVAILABLE:
        return None
    
    # Get API key from various sources
    if not api_key:
        if hasattr(st, 'secrets') and 'ANTHROPIC_API_KEY' in st.secrets:
            api_key = st.secrets['ANTHROPIC_API_KEY']
        elif os.getenv('ANTHROPIC_API_KEY'):
            api_key = os.getenv('ANTHROPIC_API_KEY')
    
    if not api_key:
        return None
    
    # Initialize Claude
    llm = ChatAnthropic(
        model=model,
        anthropic_api_key=api_key,
        temperature=0.3,
        max_tokens=2000
    )
    
    # Define tools for the agent
    @tool
    def parse_checklist_tool(checklist_text: str) -> Dict:
        """Parse a due diligence checklist into structured format"""
        return {"status": "parsing", "text": checklist_text[:100]}
    
    @tool
    def analyze_relevance_tool(doc_text: str, checklist_item: str) -> float:
        """Analyze how relevant a document is to a checklist item"""
        return 0.75  # Placeholder
    
    @tool
    def extract_information_tool(doc_text: str, query: str) -> str:
        """Extract specific information from a document"""
        return f"Extracted info about {query} from document"
    
    # Create the graph
    workflow = StateGraph(AgentState)
    
    # Define nodes
    def route_task(state: AgentState) -> AgentState:
        """Route to appropriate task based on current state"""
        messages = state["messages"]
        if not messages:
            return state
        
        last_message = messages[-1].content if messages else ""
        
        # Determine next action based on message content
        if "parse" in last_message.lower() and "checklist" in last_message.lower():
            state["next_action"] = TaskType.PARSE_CHECKLIST.value
        elif "analyze" in last_message.lower() or "match" in last_message.lower():
            state["next_action"] = TaskType.MATCH_CHECKLIST.value
        elif "?" in last_message:
            state["next_action"] = TaskType.ANSWER_QUESTION.value
        else:
            state["next_action"] = TaskType.SUMMARIZE_FINDINGS.value
        
        return state
    
    def parse_checklist_node(state: AgentState) -> AgentState:
        """Parse checklist using Claude"""
        messages = state["messages"]
        checklist_text = messages[-1].content if messages else ""
        
        prompt = f"""Parse this due diligence checklist into a structured JSON format.
        
Extract categories (A., B., C.) and numbered items.

Return ONLY valid JSON:
{{
    "A": {{
        "name": "Category Name",
        "items": [{{"text": "item", "number": 1}}]
    }}
}}

Checklist:
{checklist_text[:3000]}

JSON:"""
        
        response = llm.invoke([HumanMessage(content=prompt)])
        
        try:
            # Parse JSON from response
            json_str = response.content
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0]
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0]
            
            parsed = json.loads(json_str.strip())
            state["checklist"] = parsed
            state["messages"].append(AIMessage(content=f"Parsed {len(parsed)} categories"))
        except Exception as e:
            state["messages"].append(AIMessage(content=f"Parsing failed: {str(e)}"))
        
        return state
    
    def match_checklist_node(state: AgentState) -> AgentState:
        """Match documents to checklist items"""
        checklist = state.get("checklist", {})
        documents = state.get("documents", [])
        
        if not checklist or not documents:
            state["messages"].append(AIMessage(content="Need both checklist and documents to match"))
            return state
        
        # For each checklist item, find relevant documents
        findings = {}
        for cat_letter, category in checklist.items():
            cat_findings = []
            for item in category.get("items", []):
                # Use Claude to assess relevance
                prompt = f"""Which of these documents is relevant to: {item['text']}
                
Documents: {[d.get('name', 'Unknown') for d in documents[:10]]}

List the relevant document names only."""
                
                response = llm.invoke([HumanMessage(content=prompt)])
                cat_findings.append({
                    "item": item['text'],
                    "relevant_docs": response.content
                })
            
            findings[category['name']] = cat_findings
        
        state["findings"] = findings
        state["messages"].append(AIMessage(content=f"Matched checklist to {len(documents)} documents"))
        
        return state
    
    def answer_question_node(state: AgentState) -> AgentState:
        """Answer questions using document context"""
        messages = state["messages"]
        question = messages[-1].content if messages else ""
        documents = state.get("documents", [])
        
        # Create context from documents
        context = "\n".join([f"- {d.get('name', 'Unknown')}: {d.get('text', '')[:200]}" 
                            for d in documents[:5]])
        
        prompt = f"""Answer this question based on the documents:

Question: {question}

Document Context:
{context}

Provide a comprehensive answer with citations."""
        
        response = llm.invoke([HumanMessage(content=prompt)])
        state["messages"].append(AIMessage(content=response.content))
        
        return state
    
    def summarize_node(state: AgentState) -> AgentState:
        """Summarize findings"""
        findings = state.get("findings", {})
        
        if not findings:
            state["messages"].append(AIMessage(content="No findings to summarize"))
            return state
        
        prompt = f"""Provide an executive summary of the due diligence findings:

{json.dumps(findings, indent=2)[:2000]}

Focus on:
1. Completeness of documentation
2. Key gaps or concerns
3. Overall assessment"""
        
        response = llm.invoke([HumanMessage(content=prompt)])
        state["messages"].append(AIMessage(content=response.content))
        
        return state
    
    # Add nodes to workflow
    workflow.add_node("route", route_task)
    workflow.add_node("parse_checklist", parse_checklist_node)
    workflow.add_node("match_checklist", match_checklist_node)
    workflow.add_node("answer_question", answer_question_node)
    workflow.add_node("summarize", summarize_node)
    
    # Define edges
    workflow.set_entry_point("route")
    
    # Conditional routing based on next_action
    def route_condition(state: AgentState) -> str:
        next_action = state.get("next_action")
        if next_action == TaskType.PARSE_CHECKLIST.value:
            return "parse_checklist"
        elif next_action == TaskType.MATCH_CHECKLIST.value:
            return "match_checklist"
        elif next_action == TaskType.ANSWER_QUESTION.value:
            return "answer_question"
        else:
            return "summarize"
    
    workflow.add_conditional_edges(
        "route",
        route_condition,
        {
            "parse_checklist": "parse_checklist",
            "match_checklist": "match_checklist",
            "answer_question": "answer_question",
            "summarize": "summarize"
        }
    )
    
    # All task nodes go to END
    workflow.add_edge("parse_checklist", END)
    workflow.add_edge("match_checklist", END)
    workflow.add_edge("answer_question", END)
    workflow.add_edge("summarize", END)
    
    # Compile with memory
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    return app, llm


def batch_summarize_documents(documents: List[Dict], llm: ChatAnthropic, batch_size: int = 5) -> List[Dict]:
    """
    Summarize documents using LangChain's built-in batch processing for true parallelization.
    Returns documents with added 'summary' field.
    """
    
    def create_prompt_for_doc(doc: Dict) -> str:
        """Create a summarization prompt for a single document"""
        # Extract text preview (first 1000 chars)
        text_preview = doc.get('content', '')[:1000] if doc.get('content') else ''
        doc_name = doc.get('name', 'Unknown')
        doc_path = doc.get('path', '')
        
        return f"""Create a brief 1-2 sentence summary of what this document represents.
Focus on the document's purpose and key content.

Document: {doc_name}
Path: {doc_path}
Content preview:
{text_preview}

Summary (1-2 sentences only):"""
    
    # Process documents in batches
    summarized_docs = []
    total_docs = len(documents)
    total_batches = (total_docs + batch_size - 1) // batch_size
    
    for batch_num, i in enumerate(range(0, total_docs, batch_size), 1):
        batch = documents[i:i + batch_size]
        batch_end = min(i + batch_size, total_docs)
        
        # Update progress with batch info
        if hasattr(st, 'progress') and 'summary_progress' in st.session_state:
            progress = i / total_docs
            st.session_state.summary_progress.progress(
                progress, 
                text=f"📝 Processing batch {batch_num}/{total_batches} (docs {i+1}-{batch_end} of {total_docs})"
            )
        
        # Create prompts for all documents in the batch
        prompts = [create_prompt_for_doc(doc) for doc in batch]
        
        # Convert prompts to HumanMessage format for batch processing
        messages_batch = [[HumanMessage(content=prompt)] for prompt in prompts]
        
        try:
            # Use LangChain's batch method for parallel processing
            # Increase concurrency for larger batches
            max_concurrent = min(batch_size, 10)  # Cap at 10 concurrent requests
            responses = llm.batch(
                messages_batch, 
                config={"max_concurrency": max_concurrent}
            )
            
            # Extract summaries from responses
            batch_summaries = [response.content.strip() if response else f"Document: {doc.get('name', 'Unknown')}" 
                              for response, doc in zip(responses, batch)]
        except Exception as e:
            # Fallback to sequential processing if batch fails
            print(f"Batch {batch_num} processing failed: {e}. Falling back to sequential.")
            batch_summaries = []
            for doc_idx, doc in enumerate(batch):
                prompt = create_prompt_for_doc(doc)
                try:
                    response = llm.invoke([HumanMessage(content=prompt)])
                    batch_summaries.append(response.content.strip())
                except Exception as inner_e:
                    print(f"Failed to summarize {doc.get('name', 'Unknown')}: {inner_e}")
                    batch_summaries.append(f"Document: {doc.get('name', 'Unknown')}")
                
                # Update progress within fallback
                if hasattr(st, 'progress') and 'summary_progress' in st.session_state:
                    sub_progress = (i + doc_idx + 1) / total_docs
                    st.session_state.summary_progress.progress(
                        sub_progress,
                        text=f"📝 Sequential fallback: {i + doc_idx + 1}/{total_docs}"
                    )
        
        # Add summaries to documents
        for doc, summary in zip(batch, batch_summaries):
            doc['summary'] = summary
            summarized_docs.append(doc)
        
        # Small delay between batches to avoid rate limits
        if batch_num < total_batches:
            import time
            time.sleep(0.5)  # 500ms delay between batches
    
    return summarized_docs


def create_document_embeddings_with_summaries(documents: List[Dict], model) -> Dict[str, Any]:
    """
    Create embeddings for documents using their LLM-generated summaries.
    Returns a dict with document info and embeddings.
    """
    doc_embeddings = []
    doc_info = []
    
    for doc in documents:
        # Combine filename, path context, and LLM summary for rich embedding
        doc_name = doc.get('name', 'Unknown')
        doc_path = doc.get('path', '')
        summary = doc.get('summary', '')
        
        # Create rich text representation
        embedding_text = f"{doc_name}\n{doc_path}\n{summary}"
        
        # Generate embedding
        embedding = model.encode(embedding_text)
        
        doc_embeddings.append(embedding)
        doc_info.append({
            'name': doc_name,
            'path': doc_path,
            'summary': summary,
            'embedding_text': embedding_text,
            'original_doc': doc
        })
    
    return {
        'embeddings': doc_embeddings,
        'documents': doc_info
    }


def match_checklist_with_summaries(
    checklist: Dict, 
    doc_embeddings_data: Dict,
    model,
    threshold: float = 0.35
) -> Dict:
    """
    Match checklist items against document summaries using embeddings.
    """
    import numpy as np
    
    doc_embeddings = np.array(doc_embeddings_data['embeddings'])
    doc_info = doc_embeddings_data['documents']
    
    results = {}
    
    for cat_letter, category in checklist.items():
        cat_name = category.get('name', '')
        cat_results = {
            'name': cat_name,
            'letter': cat_letter,
            'total_items': len(category.get('items', [])),
            'matched_items': 0,
            'items': []
        }
        
        for item in category.get('items', []):
            item_text = item.get('text', '')
            
            # Create embedding for checklist item with category context
            checklist_embedding_text = f"{cat_name}: {item_text}"
            item_embedding = model.encode(checklist_embedding_text)
            
            # Calculate similarities with all documents
            similarities = np.dot(doc_embeddings, item_embedding) / (
                np.linalg.norm(doc_embeddings, axis=1) * np.linalg.norm(item_embedding)
            )
            
            # Find matching documents above threshold
            matches = []
            for idx, sim in enumerate(similarities):
                if sim > threshold:
                    matches.append({
                        'name': doc_info[idx]['name'],
                        'path': doc_info[idx]['path'],
                        'summary': doc_info[idx]['summary'],
                        'score': float(sim),
                        'metadata': doc_info[idx].get('original_doc', {}).get('metadata', {})
                    })
            
            # Sort by score and keep top matches
            matches = sorted(matches, key=lambda x: x['score'], reverse=True)[:5]
            
            item_result = {
                'text': item_text,
                'original': item.get('original', item_text),
                'matches': matches
            }
            
            if matches:
                cat_results['matched_items'] += 1
            
            cat_results['items'].append(item_result)
        
        results[cat_letter] = cat_results
    
    return results


class DDChecklistAgent:
    """High-level interface for the LangGraph agent"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-haiku-20240307"):
        result = get_langgraph_agent(api_key, model)
        if result:
            self.app, self.llm = result
            self.thread_id = "dd-checklist-session"
        else:
            self.app = None
            self.llm = None
    
    def is_available(self) -> bool:
        return self.app is not None
    
    def parse_checklist(self, checklist_text: str) -> Optional[Dict]:
        """Parse checklist using the agent"""
        if not self.app:
            return None
        
        try:
            # Run the agent
            result = self.app.invoke(
                {"messages": [HumanMessage(content=f"Parse this checklist: {checklist_text}")]},
                config={"configurable": {"thread_id": self.thread_id}}
            )
            
            return result.get("checklist")
        except Exception as e:
            st.error(f"Agent error: {str(e)}")
            return None
    
    def match_documents(self, checklist: Dict, documents: List[Dict]) -> Dict:
        """Match documents to checklist items"""
        if not self.app:
            return {}
        
        try:
            # Prepare state
            initial_state = {
                "messages": [HumanMessage(content="Match documents to checklist items")],
                "checklist": checklist,
                "documents": documents,
                "findings": {}
            }
            
            result = self.app.invoke(
                initial_state,
                config={"configurable": {"thread_id": self.thread_id}}
            )
            
            return result.get("findings", {})
        except Exception as e:
            st.error(f"Agent error: {str(e)}")
            return {}
    
    def answer_question(self, question: str, documents: List[Dict]) -> str:
        """Answer a question using document context"""
        if not self.app:
            return "Agent not available"
        
        try:
            initial_state = {
                "messages": [HumanMessage(content=question)],
                "documents": documents
            }
            
            result = self.app.invoke(
                initial_state,
                config={"configurable": {"thread_id": self.thread_id}}
            )
            
            # Get the last AI message
            messages = result.get("messages", [])
            for msg in reversed(messages):
                if isinstance(msg, AIMessage):
                    return msg.content
            
            return "No answer generated"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def summarize_findings(self, findings: Dict) -> str:
        """Generate executive summary"""
        if not self.app:
            return "Agent not available"
        
        try:
            initial_state = {
                "messages": [HumanMessage(content="Summarize the due diligence findings")],
                "findings": findings
            }
            
            result = self.app.invoke(
                initial_state,
                config={"configurable": {"thread_id": self.thread_id}}
            )
            
            # Get the last AI message
            messages = result.get("messages", [])
            for msg in reversed(messages):
                if isinstance(msg, AIMessage):
                    return msg.content
            
            return "No summary generated"
        except Exception as e:
            return f"Error: {str(e)}"

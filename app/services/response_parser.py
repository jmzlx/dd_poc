#!/usr/bin/env python3
"""
Response Parser

Handles response parsing and formatting functions for AI operations.
Provides methods for creating prompts and processing AI responses.
"""

from typing import List, Dict, Any, Optional

from app.core.exceptions import ProcessingError


class ResponseParser:
    """
    Parser for AI responses and prompt generation.

    This class provides methods for creating structured prompts
    and processing AI responses for different analysis types.
    """

    @staticmethod
    def create_overview_prompt(
        context_docs: List[str],
        strategy_text: Optional[str],
        checklist_results: Optional[Dict]
    ) -> str:
        """Create overview analysis prompt"""
        prompt = "Based on the following company documents, provide a comprehensive overview analysis:\n\n"

        if context_docs:
            prompt += "Company Documents:\n" + "\n\n".join(context_docs) + "\n\n"

        if strategy_text:
            prompt += f"Strategic Context:\n{strategy_text[:1000]}\n\n"

        if checklist_results:
            prompt += f"Checklist Findings:\n{str(checklist_results)[:1000]}\n\n"

        prompt += """Please provide:
1. Company overview and business model
2. Key strengths and competitive advantages
3. Main risks and challenges
4. Financial health indicators
5. Strategic recommendations

Be specific, factual, and focus on the most important insights."""

        return prompt

    @staticmethod
    def create_strategic_prompt(
        context_docs: List[str],
        strategy_text: Optional[str],
        checklist_results: Optional[Dict]
    ) -> str:
        """Create strategic analysis prompt"""
        prompt = "Provide a strategic analysis based on the following company information:\n\n"

        if strategy_text:
            prompt += f"Strategic Framework:\n{strategy_text[:1000]}\n\n"

        if context_docs:
            prompt += "Company Documents:\n" + "\n\n".join(context_docs) + "\n\n"

        if checklist_results:
            prompt += f"Operational Findings:\n{str(checklist_results)[:1000]}\n\n"

        prompt += """Please analyze:
1. Strategic positioning and market opportunities
2. Operational strengths and weaknesses
3. Risk mitigation strategies
4. Growth potential and recommendations
5. Investment considerations

Focus on strategic implications and actionable insights."""

        return prompt

    @staticmethod
    def create_checklist_prompt(context_docs: List[str]) -> str:
        """Create checklist analysis prompt"""
        prompt = "Analyze the following documents against standard due diligence checklist items:\n\n"

        if context_docs:
            prompt += "Documents to Analyze:\n" + "\n\n".join(context_docs) + "\n\n"

        prompt += """For each major due diligence category, identify:
1. What information is available in the documents
2. What information appears to be missing
3. Any red flags or concerns identified
4. Recommendations for further investigation

Be thorough and specific in your analysis."""

        return prompt

    @staticmethod
    def create_questions_prompt(context_docs: List[str]) -> str:
        """Create questions analysis prompt"""
        prompt = "Answer due diligence questions based on the following documents:\n\n"

        if context_docs:
            prompt += "Reference Documents:\n" + "\n\n".join(context_docs) + "\n\n"

        prompt += """For each question, provide:
1. Direct answer based on available information
2. Supporting evidence from the documents
3. Confidence level in the answer
4. Any additional context or caveats

If information is not available, clearly state this and suggest what additional information would be needed."""

        return prompt

    @staticmethod
    def create_question_answer_prompt(question: str, context_docs: List[str]) -> str:
        """Create prompt for answering a specific question"""
        return f"""Based on the following document excerpts, please answer this question:

Question: {question}

Relevant Document Excerpts:
{"\n\n".join(context_docs[:5])}

Please provide a comprehensive, factual answer with specific references to the source documents.
If the information is not available in the provided context, clearly state this."""

    @staticmethod
    def format_response(response: str, max_length: Optional[int] = None) -> str:
        """
        Format and clean AI response.

        Args:
            response: Raw AI response
            max_length: Optional maximum length for the response

        Returns:
            Formatted response
        
        Raises:
            ProcessingError: If response formatting fails
        """
        try:
            if not response:
                raise ValueError("Response cannot be empty")

            result = response.strip()
            if max_length and len(result) > max_length:
                result = result[:max_length] + "..."
            return result
        except Exception as e:
            raise ProcessingError(f"Failed to format AI response: {e}")

    @staticmethod
    def prepare_context_documents(documents: Dict[str, Dict[str, Any]], max_docs: int = 5) -> List[str]:
        """
        Prepare context documents for AI processing.

        Args:
            documents: Dictionary mapping document names to document data
            max_docs: Maximum number of documents to process

        Returns:
            List of formatted document contexts
            
        Raises:
            ProcessingError: If document preparation fails
        """
        try:
            if not documents:
                raise ValueError("No documents provided for context preparation")

            context_docs = []
            for doc_key, doc_data in list(documents.items())[:max_docs]:
                if isinstance(doc_data, dict) and 'content' in doc_data:
                    content = doc_data['content'][:1000]  # Truncate long content
                    context_docs.append(f"Document: {doc_data.get('name', doc_key)}\n{content}")

            if not context_docs:
                raise ValueError("No valid documents found with content")

            return context_docs
        except Exception as e:
            raise ProcessingError(f"Failed to prepare context documents: {e}")

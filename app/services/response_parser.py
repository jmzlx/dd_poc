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
        """Create overview analysis prompt focused on target company perspective"""
        prompt = "Analyze the following target company documents from an acquisition perspective:\n\n"

        if context_docs:
            prompt += "Target Company Documents:\n" + "\n\n".join(context_docs) + "\n\n"

        if strategy_text:
            prompt += f"Acquirer's Strategic Context (for reference):\n{strategy_text[:1000]}\n\n"

        if checklist_results:
            prompt += f"Due Diligence Findings:\n{str(checklist_results)[:1000]}\n\n"

        prompt += """Please provide a comprehensive analysis of the TARGET COMPANY focusing on:

1. **Company Overview**: Business model, market position, and core operations of the target
2. **Strategic Value**: Why this target company would be attractive for acquisition
3. **Competitive Strengths**: Key assets, capabilities, and competitive advantages the target brings
4. **Risk Assessment**: Main operational, financial, and strategic risks associated with the target
5. **Financial Health**: Target company's financial position and performance indicators
6. **Acquisition Rationale**: How the target fits acquisition criteria and strategic objectives

Focus on analyzing the target company as a potential acquisition candidate. Be specific, factual, and highlight both opportunities and concerns from an acquirer's due diligence perspective."""

        return prompt

    @staticmethod
    def create_strategic_prompt(
        context_docs: List[str],
        strategy_text: Optional[str],
        checklist_results: Optional[Dict]
    ) -> str:
        """Create strategic analysis prompt focused on target company from acquisition perspective"""
        prompt = "Conduct a strategic analysis of the target company from an acquisition perspective:\n\n"

        if strategy_text:
            prompt += f"Acquirer's Strategic Framework (for context):\n{strategy_text[:1000]}\n\n"

        if context_docs:
            prompt += "Target Company Documents:\n" + "\n\n".join(context_docs) + "\n\n"

        if checklist_results:
            prompt += f"Due Diligence Findings:\n{str(checklist_results)[:1000]}\n\n"

        prompt += """Please provide a strategic analysis of the TARGET COMPANY focusing on:

1. **Strategic Fit Assessment**: How well the target aligns with the acquirer's strategic objectives and portfolio
2. **Market Position Analysis**: Target's competitive position, market share, and industry dynamics
3. **Value Creation Opportunities**: Potential synergies, cross-selling opportunities, and operational improvements
4. **Integration Considerations**: Key challenges and opportunities for successful integration
5. **Risk-Adjusted Valuation**: Strategic risks, regulatory concerns, and market vulnerabilities
6. **Post-Acquisition Strategy**: Recommended approach for maximizing value creation after acquisition

Analyze the target company as an acquisition candidate, evaluating both strategic alignment and value creation potential. Consider the acquirer's strategic framework when assessing fit and synergy opportunities."""

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

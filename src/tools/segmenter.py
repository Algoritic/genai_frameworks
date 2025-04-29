import json
import re
from typing import Dict, List, Tuple

from llms.llm import LLMBase
from core.logger import logger
from pydantic import BaseModel


class Segment(BaseModel):
    document_name: str
    sub_document_name: str
    confidence: float
    reasoning: str


class DocumentSegmenter:

    def __init__(self,
                 pages: List[str],
                 llm: LLMBase,
                 confidence_threshold: float = 0.7,
                 min_document_pages: int = 1,
                 max_document_pages: int = 100):
        self.pages = pages
        self.min_document_pages = min_document_pages
        self.max_document_pages = max_document_pages
        self.llm_service = llm
        self.confidence_threshold = confidence_threshold

    def _extract_representative_content(self, page_content: str) -> str:
        """
        Extract key portions of the page that help identify document boundaries.

        Args:
            page_content: Full text content of a page

        Returns:
            Formatted string with key content sections
        """
        # This is a simplified implementation
        # In a production environment, you would:
        # - Extract header areas based on layout analysis
        # - Use NER or pattern matching to identify metadata
        # - Consider visual layout information

        lines = page_content.split('\n')
        page_length = len(lines)

        # Extract different sections of the page
        header_section = '\n'.join(lines[:min(5, page_length // 5)])

        # Beginning and ending content - always up to 20 lines
        beginning_content = '\n'.join(lines[:min(20, page_length)])

        ending_content = '\n'.join(lines[max(0, page_length - 20):])

        # Combine with appropriate context
        representative_content = f"""
        PAGE HEADER:
        {header_section}

        BEGINNING CONTENT:
        {beginning_content}

        ENDING CONTENT:
        {ending_content}
        """

        return representative_content

    def _construct_boundary_detection_prompt(self, prev_page_doc_name: str,
                                             prev_page_sub_doc_name: str,
                                             curr_page_content: str,
                                             page_number: int) -> str:
        """
        Construct the prompt for boundary detection.

        Args:
            prev_page_content: Content from previous page
            curr_page_content: Content from current page
            page_number: Current page number

        Returns:
            Formatted prompt string
        """

        prompt = f"""
        Your task is to determine the one most relevant document name and sub document name for the given page content.

        === PREVIOUS PAGE (DOCUMENT NAME,SUB DOCUMENT NAME)  ===
        {prev_page_doc_name}

        There is a high chance that the document name of the current page is the same as the previous page. You must provide strong justification if you think the document name is different.

        === CURRENT PAGE CONTENT ===
        {curr_page_content}

        Below are the rules for document tagging. Please follow the descriptions strictly. Use only the specified keywords. Include keywords in your justification.
        DO NOT HALLUCINATE OR MAKE UP ANYTHING. DO NOT ADD ANYTHING THAT IS NOT IN THE KEYWORDS.

       "document_name,sub_document_name,description,keywords
Advice of release,Advise of Release,"IIn short AOR. Letter Head Lawyer. Advice of Release can contain Land Search keyword. Letter head contains law firm name. This is a checklist for all the document type below. This document should contain EXPLICIT keywords from other classes. Lawyer confirmation on the completeness of the document by using checklist. Instruction to release the loan.","eLDS Lawyer Checklist', 'Advise of Release','1. Security Documents', '2. Searches', 'a.Land search', 'b(i) & b(ii) Bankruptcy Search', '3. Letter of Undertaking/ Disclaimer/ Consent Page', '4. Others', '5. The Related Documents', '6. Comments (If Any)', 'release …', 'Your faithfully', 'Withdrawal of private caveat', 'Charge', 'Letter of Guarantee', and 'Memorandum of Deposit cum Assignment', 'Important Note', 'Digital documents, use of electronic signature or digital signature and delivery of documents electronically'"
Bankruptcy Search Result / Company Winding Up Search Result,Bankruptcy Search Result / Company Winding Up Search Result,"Letter from Jabatan Insolvensi Malaysia / Malaysian Department of Insolvency","'Bankruptcy Search Result', 'Company Winding Up Search Result'"
Statutory Declaration (Owner Occupancy), Statutory Declaration (Owner Occupancy), "Letter with word "Declaration" or "Form of Declaration" & special keyword :…used for my/our own business…", "Declaration', 'Form of Declaration', '...used for my own business', '...used for our own business...'"
Statutory Declaration (Owner Occupancy),Stamp Certificate,"Letter from Lembaga Hasil Dalam Negeri. Type of Instrument must be AKUAN BERKANUN. This letter can be before or after the Statutory Declaration.","'Type of Instrument','Sijil Setem / Stamp Certificate','AKAUN BERKANUN'"
Acknowledgement of Legal Representation,Acknowledgement of Legal Representation,"Letter signed by Purchaser / Borrower", "'Acknowledgement of Legal Representation'"
Land Search,Land Search,""Letter with Keyword "Letter from State Land Office", "'Carian Persendirian', 'Carian Rasmi'"
Confirmation Letter from Pajabat Tanah dan Galian,Confirmation Letter from Pajabat Tanah dan Galian,Letter Head from State Land Office / Pajabat Tanah dan Galian Negeri,""
Confirmation Letter from Perbadanan Kemajuan Perumahan Negeri,Confirmation Letter from Perbadanan Kemajuan Perumahan Negeri, Letter Head from Perbadanan Kemajuan Perumahan Negeri,"'Perbadanan Kemajuan Perumahan Negeri'"
Quit Rent,Quit Rent,"Letter from State Land Office / Pajabat Tanah dan Galian Negeri and detail, have amount stated", "'Title No','Mukim', 'Daerah', 'amount'"
Full Power of Attorney,Full Power of Attorney,"Document with word "Full Power of Attorney" with Donor & Attorney name", "'Full Power of Attorney'"
Solicitor Letter of Undertaking,Solicitor Letter of Undertaking,"Letter from Lawyer address to Maybank", "'undertake…'"
Redemption Statement,Redemption Statement,"Letter from Bank address to Maybank","'Redemption Statement'"
Developer Letter of Undertaking,Developer Letter of Undertaking,"Developer's Letter address to Maybank.","'Letter of Undertaking', 'the difference between the Purchase Price and Financial Facility has ….', 'currently charge'"
Letter of Offer,Stamp Certificate,"Letter from Lembaga Hasil Dalam Negeri. Type of Instrument must be LETTER OF OFFER. This letter can be before or in between the Letter of Offer.","'Sijil Setem', 'Stamp Certificate', 'Type of Instrument', 'LETTER OF OFFER'"
Letter of Offer,Letter of Offer,"Letter with Bank Letter Head", "'Type of Facility', 'Facility Amount', 'Acceptance Page', 'Acknowledgement Page', 'Annexure 1', 'Specific Terms & Conditions', 'Annexure 2 - Part A', 'Annexure 2 - Part B', 'Annexure 2', 'EXPRESS RIGHT TO CANCEL'"
Notice of Assignment,Notice of Assignment,"Letter with keyword 'Notice of Assignment", "'Notice of Assignment'"
Presentation Receipt,Presentation Receipt,Receipt from Pejabat Ketua Pendaftar Mahkamah Persekutuan Malaysia, "'Presentation Receipt'"
Deed of Assignment,Deed of Assignment,"Document with word "Deed of Assignment" with Maybank Name and Borrower(S) Name", "'Deed of Assignment', 'Execution Page - The Bank' , 'Execution Page - The Assignor', 'Certificate of Understanding', 'Authentication of Power of Attorney','First Schedule', 'First Schedule - Sector No.1',''First Schedule - Sector No.2', 'First Schedule - Sector No.3', 'First Schedule - Sector No.4', ''First Schedule - Sector No.5', ''First Schedule - Sector No.6'"
Deed of Assignment,Stamp Certificate,"Letter from Lembaga Hasil Dalam Negeri. Type of Instrument must be Deed of Assignment. This letter can be before or in between Deed of Assignment.","Type of Instrument','Sijil Setem / Stamp Certificate','Deed of Assignment'"
Sale & Purchase Agreement,Sale & Purchase Agreement,"Document with word Sale & Purchase Agreement with information at least 2 parties e.g Seller/Developer and Purchaser name.","Seller', 'Developer', 'Purchaser', 'Schedule G', 'Schedule H', 'Preamble', 'charged to...', 'Housing Development Account','Signing Page - In Witness Whereof'"
Sale & Purchase Agreement,Stamp Certificate,"Letter from Lembaga Hasil Dalam Negeri. Type of Instrument must be SALES AND PURCHASE AGREEMENT. This letter can be before or in between the Sales and Purchase Agreement", "'Sijil Setem', 'Stamp Certificate', 'Type of Instrument', 'SALES AND PURCHASE AGREEMENT'"
Commodity Murabahah Facility Agreement,Commodity Murabahah Facility Agreement,"Document with Maybank Name and Borrower(S) Name. This Document may consist set of Letter of Offer.Page First Schedule No.1 - No.8. Last Page CMFA - Understanding", "Commodity Murabahah Facility Agreement', 'Borrower(s) Name', 'Schedule No.1',Schedule No.2','Schedule No.3','Schedule No.4'.'Schedule No.5','Schedule No.6','Schedule No.7','Schedule No.8','CMFA - Understanding'"
Commodity Murabahah Facility Agreement,Stamp Certificate,"Letter from Lembaga Hasil Dalam Negeri with Type of Instrument. Type of Instrument must be Commodity Murabahah Facility Agreement. This letter can be before or in between Commodity Murabahah Facility Agreement.","'Sijil Setem', 'Stamp Certificate', 'Type of Instrument', 'Commodity Murabahah Facility Agreement'"
        """

        return prompt

    def _parse_boundary_response(self,
                                 llm_response: str) -> Tuple[bool, float, str]:
        """
        Parse the LLM response to extract decision, confidence, and reasoning.

        Args:
            llm_response: Raw response from the LLM

        Returns:
            Tuple of (is_boundary: bool, confidence_score: float, reasoning: str)
        """
        # Simple parsing logic - in production, you'd want more robust parsing
        lines = llm_response.strip().split('\n')
        decision = False
        confidence_score = 0.0
        reasoning = ""

        for line in lines:
            if line.startswith("DECISION:"):
                decision_text = line.replace("DECISION:", "").strip().upper()
                # decision = decision_text == "YES"
            elif line.startswith("CONFIDENCE:"):
                confidence_text = line.replace("CONFIDENCE:",
                                               "").strip().upper()
                if confidence_text == "HIGH":
                    confidence_score = 0.9
                elif confidence_text == "MEDIUM":
                    confidence_score = 0.7
                elif confidence_text == "LOW":
                    confidence_score = 0.4
            elif line.startswith("REASONING:"):
                reasoning = line.replace("REASONING:", "").strip()
                # Collect any additional reasoning lines
                reasoning_start_idx = lines.index(line)
                if reasoning_start_idx < len(lines) - 1:
                    additional_reasoning = '\n'.join(
                        lines[reasoning_start_idx + 1:])
                    reasoning = f"{reasoning} {additional_reasoning}"

        return decision_text, confidence_score, reasoning

    def _detect_boundary(self, prev_page_document_name: str,
                         prev_page_sub_document_name: str,
                         curr_page_content: str, page_number: int) -> Segment:
        """
        Use LLM to determine if there's a document boundary between pages.

        Args:
            prev_page_content: Content from previous page
            curr_page_content: Content from current page
            page_number: Current page number

        Returns:
            Tuple of (is_boundary, confidence_score, reasoning)
        """
        # Construct prompt for the LLM
        prompt = self._construct_boundary_detection_prompt(
            prev_page_document_name, prev_page_sub_document_name,
            curr_page_content, page_number)

        # Query the LLM (abstracted)
        llm_response = self.llm_service.generate_structured_schema(
            method="json_schema",
            prompt={
                "system":
                """You are a meticulous document-classification specialist.
Your task is to read a single, OCR'ed, full-text document (Malay and/or English) and decide which—if any—of the predefined document types it belongs to.""",
                "user": prompt
            },
            json_schema=json.loads(
                """         {
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "DocumentTagging",
  "description": "Schema for Document tagging",
  "type": "object",
  "strict": true,
  "properties": {
    "document_name": {
      "type": "string",
      "description": "The primary name of the document",
      "enum": [
        "Advice of release",
        "Bankruptcy Search Result/ Company Winding Up Search Result",
        "Statutory Declaration (Owner Occupancy)",
        "Acknowledgement of Legal Representation",
        "Land Search",
        "Confirmation Letter from Pajabat Tanah dan Galian",
        "Confirmation Letter from Perbadanan Kemajuan Perumahan Negeri",
        "Quit Rent",
        "Full Power of Attorney",
        "Solicitor Letter of Undertaking",
        "Redemption Statement",
        "Developer Letter of Undertaking",
        "Letter of Offer",
        "Notice of Assignment",
        "Presentation Receipt",
        "Deed of Assignment",
        "Sale & Purchase Agreement",
        "Commodity Murabahah Facility Agreement"
      ]
    },
    "sub_document_name": {
      "type": "string",
      "description": "The subcategory or alternate name for the document",
      "enum": [
        "Advise of Release",
        "Bankruptcy Search Result/ Company Winding Up Search Result",
        "Stamp Certificate",
        "Statutory Declaration (Owner Occupancy)",
        "Acknowledgement of Legal Representation",
        "Land Search",
        "Confirmation Letter from Pajabat Tanah dan Galian",
        "Confirmation Letter from Perbadanan Kemajuan Perumahan Negeri",
        "Quit Rent",
        "Full Power of Attorney",
        "Solicitor Letter of Undertaking",
        "Redemption Statement",
        "Developer Letter of Undertaking",
        "Letter of Offer",
        "Notice of Assignment",
        "Presentation Receipt",
        "Deed of Assignment",
        "Sale & Purchase Agreement",
        "Commodity Murabahah Facility Agreement"
      ]
    },
    "confidence":{
      "type":"number",
       "description":"The confidence level to the result, ranged from 0 to 1, can be in floating value"
    },
    "reasoning": {
      "type": "string",
      "description": "Detailed explanation or matching criteria for the document"
    }
  },
  "required": [
    "document_name",
    "sub_document_name",
    "confidence",
    "reasoning"
  ]
}""", ))
        return Segment.model_validate(
            llm_response,
            strict=True,
        )
        # Parse LLM response
        result, confidence, reasoning = self._parse_boundary_response(
            llm_response)
        # return True, 0.1, "hello"
        return result, confidence, reasoning

    def _validate_boundaries(self, boundaries: List[int],
                             total_pages: int) -> List[int]:
        """
        Apply constraints to validate and potentially adjust document boundaries.

        Args:
            boundaries: List of page numbers that start documents
            total_pages: Total number of pages in the file

        Returns:
            Validated list of boundary page numbers
        """
        if not boundaries or boundaries[0] != 1:
            boundaries.insert(0, 1)  # Ensure first page is a boundary

        # Check for documents that are too short or too long
        valid_boundaries = [boundaries[0]]

        for i in range(1, len(boundaries)):
            doc_length = boundaries[i] - valid_boundaries[-1]

            # Skip boundaries that would create documents that are too short
            if doc_length < self.min_document_pages:
                logger.warning(
                    f"Skipping boundary at page {boundaries[i]} as it would create a document "
                    f"of only {doc_length} pages")
                continue

            # Split documents that are too long
            if doc_length > self.max_document_pages:
                logger.warning(
                    f"Document starting at page {valid_boundaries[-1]} is too long "
                    f"({doc_length} pages). Splitting.")

                # Add artificial boundaries to split the document
                pages_per_section = self.max_document_pages
                start_page = valid_boundaries[-1]

                while start_page + pages_per_section < boundaries[i]:
                    next_boundary = start_page + pages_per_section
                    valid_boundaries.append(next_boundary)
                    start_page = next_boundary

            valid_boundaries.append(boundaries[i])

        # Ensure we cover all pages
        if valid_boundaries[-1] < total_pages:
            doc_length = total_pages + 1 - valid_boundaries[-1]

            # If last document would be too long, add intermediate boundaries
            if doc_length > self.max_document_pages:
                pages_per_section = self.max_document_pages
                start_page = valid_boundaries[-1]

                while start_page + pages_per_section < total_pages + 1:
                    next_boundary = start_page + pages_per_section
                    valid_boundaries.append(next_boundary)
                    start_page = next_boundary

        return valid_boundaries

    def _boundaries_to_ranges(self, boundaries: List[int],
                              total_pages: int) -> List[Dict[str, int]]:
        """
        Convert boundary page numbers to document ranges.

        Args:
            boundaries: List of page numbers that start documents
            total_pages: Total number of pages in the file

        Returns:
            List of document ranges in format [{"start": 1, "end": 10}, ...]
        """
        document_ranges = []

        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1] - 1
            document_ranges.append({"start": start, "end": end})

        # Add the last document
        if boundaries:
            start = boundaries[-1]
            end = total_pages
            document_ranges.append({"start": start, "end": end})

        return document_ranges

    def segment(self):
        """
        Segments the document into smaller parts based on the number of pages.
        """
        logger.info(f"Segmenting document with {len(self.pages)} pages")

        num_pages = len(self.pages)
        document_boundaries = [1]
        page_tag = []

        for i in range(0, num_pages):
            current_page_number = i + 1
            prev_page = self.pages[i - 1]
            current_page = self.pages[i]

            logger.info(
                f"Analyzing transition between pages {i} and {current_page_number}"
            )
            doc_entry = page_tag[i - 1] if i > 0 else ("", "")

            previous_page_document_name, previous_page_sub_document_name = doc_entry

            # Extract content samples from pages
            # prev_page_sample = self._extract_representative_content(prev_page)
            # curr_page_sample = self._extract_representative_content(
            #     current_page)

            #fallback to previous page tag if page with the word "CC" is found
            pattern = r"(?<!\w)[cC]\s*\.?\s*[cC](?=\b|[^a-zA-Z])"
            matches = re.findall(pattern, current_page)
            if len(matches) > 0:
                page_tag.insert(i, doc_entry)
                continue

            # Use LLM to determine if boundary exists
            llm_response = self._detect_boundary(
                previous_page_document_name, previous_page_sub_document_name,
                current_page, current_page_number)

            logger.info(
                f"Boundary detection result: {llm_response.document_name, llm_response.sub_document_name}, confidence: {llm_response.confidence}"
            )
            logger.debug(f"Reasoning: {llm_response.reasoning}")

            # if llm_response.document_name and llm_response.confidence >= self.confidence_threshold:
            # document_boundaries.append(current_page_number)
            logger.info(
                f"Detected document boundary at page {current_page_number}")
            page_tag.insert(
                i,
                (llm_response.document_name, llm_response.sub_document_name))

        # document_boundaries = self._validate_boundaries(
        #     document_boundaries, num_pages)

        # document_ranges = self._boundaries_to_ranges(document_boundaries,
        #                                              num_pages)

        logger.info(
            f"Segmentation complete. Identified {len(page_tag)} documents.")
        return page_tag

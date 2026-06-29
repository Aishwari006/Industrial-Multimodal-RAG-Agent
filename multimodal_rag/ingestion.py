"""
ingestion.py
------------
Responsible for parsing a PDF file and returning a flat list of LangChain
``Document`` objects ready to be embedded.

Strategy
~~~~~~~~
1. Extract every text block page-by-page with PyMuPDF.
2. Split long text blocks with RecursiveCharacterTextSplitter.
3. Extract every embedded image from each page (in-memory only).
4. Summarise each image with the local VLM (ImageSummarizer).
5. Wrap summaries as Documents with ``content_type = "image_summary"``.

Nothing is written to disk; no raw image data or base64 strings are stored.
"""

import logging
import uuid
from pathlib import Path
from typing import Generator, List

import fitz  # PyMuPDF
from PIL import Image
from io import BytesIO
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from .config import RAGConfig
from .image_summarizer import ImageSummarizer

logger = logging.getLogger(__name__)


class PDFIngester:
    """
    Parses a PDF and produces LangChain Documents (text + image summaries).

    Parameters
    ----------
    config : RAGConfig
        Pipeline configuration.
    """

    def __init__(self, config: RAGConfig) -> None:
        self._config = config
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )
        self._summarizer = ImageSummarizer(config)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def ingest(self, pdf_path: str | Path) -> List[Document]:
        """
        Parse *pdf_path* and return all Documents ready for embedding.

        Parameters
        ----------
        pdf_path : str or Path
            Absolute or relative path to the PDF file.

        Returns
        -------
        List[Document]
            Mixed list of text-chunk and image-summary Documents, each
            carrying rich metadata.

        Raises
        ------
        FileNotFoundError
            If *pdf_path* does not exist.
        RuntimeError
            If PyMuPDF cannot open the file.
        """
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")

        logger.info("Ingesting PDF: %s", path.name)
        documents: List[Document] = []

        try:
            pdf = fitz.open(str(path))
        except Exception as exc:
            raise RuntimeError(f"Cannot open PDF '{path}': {exc}") from exc

        with pdf:
            total_pages = len(pdf)
            logger.info("PDF has %d page(s).", total_pages)

            for page_index in range(total_pages):
                page = pdf[page_index]
                page_number = page_index + 1  # 1-based for humans

                # ── Text ────────────────────────────────────────────────
                documents.extend(
                    self._extract_text_documents(page, path.name, page_number)
                )

                # ── Images ──────────────────────────────────────────────
                documents.extend(
                    self._extract_image_documents(pdf, page, path.name, page_number)
                )

        logger.info(
            "Ingestion complete: %d document chunks produced from '%s'.",
            len(documents),
            path.name,
        )
        return documents

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _extract_text_documents(
        self,
        page: fitz.Page,
        source: str,
        page_number: int,
    ) -> List[Document]:
        """Extract and split text from a single PDF page."""
        raw_text: str = page.get_text("text").strip()
        if not raw_text:
            return []

        chunks = self._splitter.split_text(raw_text)
        docs: List[Document] = []
        for chunk in chunks:
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": source,
                        "page_number": page_number,
                        "content_type": "text",
                        "element_id": str(uuid.uuid4()),
                        "bounding_box": None,
                    },
                )
            )
        return docs

    def _extract_image_documents(
        self,
        pdf: fitz.Document,
        page: fitz.Page,
        source: str,
        page_number: int,
    ) -> List[Document]:
        """Extract images from a page, summarise them, return summary Documents."""
        image_list = page.get_images(full=True)
        if not image_list:
            return []

        docs: List[Document] = []
        for img_index, img_info in enumerate(image_list):
            xref = img_info[0]
            bbox = page.get_image_bbox(img_info)  # fitz.Rect or None

            try:
                base_image = pdf.extract_image(xref)
                image_bytes: bytes = base_image["image"]
                pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
            except Exception as exc:
                logger.warning(
                    "Could not decode image %d on page %d: %s",
                    img_index + 1,
                    page_number,
                    exc,
                )
                continue

            logger.debug(
                "Summarising image %d/%d on page %d …",
                img_index + 1,
                len(image_list),
                page_number,
            )
            summary = self._summarizer.summarize(pil_image, page_number)

            docs.append(
                Document(
                    page_content=summary,
                    metadata={
                        "source": source,
                        "page_number": page_number,
                        "content_type": "image_summary",
                        "element_id": str(uuid.uuid4()),
                        "bounding_box": (
                            {
                                "x0": bbox.x0,
                                "y0": bbox.y0,
                                "x1": bbox.x1,
                                "y1": bbox.y1,
                            }
                            if bbox
                            else None
                        ),
                    },
                )
            )

        return docs

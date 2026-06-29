"""
image_summarizer.py
-------------------
Converts in-memory PIL Images into text summaries using a local Ollama VLM.

No images are written to disk. The base64 encoding is used only as a transient
transport format for the Ollama vision API; it is never stored anywhere.
"""

import base64
import logging
from io import BytesIO

from PIL import Image
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

from .config import RAGConfig

logger = logging.getLogger(__name__)


class ImageSummarizer:
    """
    Wraps a local Ollama vision-capable model to generate text summaries of images.

    Parameters
    ----------
    config : RAGConfig
        Pipeline configuration (model name, base URL, prompt …).
    """

    def __init__(self, config: RAGConfig) -> None:
        self._config = config
        self._llm = ChatOllama(
            model=config.chat_model,
            base_url=config.ollama_base_url,
            temperature=0.2,
        )

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def summarize(self, image: Image.Image, page_number: int) -> str:
        """
        Generate a text description of *image*.

        Parameters
        ----------
        image       : PIL.Image.Image – the image to describe.
        page_number : int             – used only for logging.

        Returns
        -------
        str
            A detailed text summary of the image content.
        """
        b64 = self._pil_to_base64(image)
        try:
            message = HumanMessage(
                content=[
                    {"type": "text", "text": self._config.image_summary_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                ]
            )
            response = self._llm.invoke([message])
            summary: str = response.content.strip()
            logger.debug("Summarised image on page %d (%d chars).", page_number, len(summary))
            return summary
        except Exception as exc:
            logger.warning("Failed to summarise image on page %d: %s", page_number, exc)
            return f"[Image on page {page_number} — summarisation failed: {exc}]"

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _pil_to_base64(image: Image.Image) -> str:
        """Convert a PIL Image to a base64-encoded PNG string (in-memory only)."""
        buf = BytesIO()
        # Ensure RGB so PNG encoding never fails on palette/RGBA images
        rgb = image.convert("RGB")
        rgb.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

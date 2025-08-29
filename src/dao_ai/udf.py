import mimetypes
import re
import warnings
from io import BytesIO
from typing import Iterator

import pandas as pd
import pyspark.sql.functions as F
import pyspark.sql.types as T
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import (
    ConversionResult,
    DocumentConverter,
    PdfFormatOption,
    WordFormatOption,
)
from docling.pipeline.simple_pipeline import SimplePipeline
from docling_core.types.doc.document import DoclingDocument

pipeline_options: PdfPipelineOptions = PdfPipelineOptions()
pipeline_options.do_ocr = False
pipeline_options.do_table_structure = True


## Custom options are now defined per format.
def document_converter() -> DocumentConverter:
    converter: DocumentConverter = DocumentConverter(  # all of the below is optional, has internal defaults.
        allowed_formats=[
            InputFormat.PDF,
            InputFormat.IMAGE,
            InputFormat.DOCX,
            InputFormat.HTML,
            InputFormat.PPTX,
        ],  # whitelist formats, non-matching files are ignored.
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,  # pipeline options go here.
                backend=PyPdfiumDocumentBackend,  # optional: pick an alternative backend
            ),
            InputFormat.DOCX: WordFormatOption(
                pipeline_cls=SimplePipeline  # default for office formats and HTML
            ),
        },
    )
    return converter


converter: DocumentConverter = document_converter()


def parse_bytes(raw_doc_contents_bytes: bytes) -> str:
    try:
        stream: BytesIO = BytesIO(raw_doc_contents_bytes)
        document_stream: DocumentStream = DocumentStream(name="foo", stream=stream)
        result: ConversionResult = converter.convert(document_stream)
        document: DoclingDocument = result.document
        markdown: str = document.export_to_markdown()
        return markdown
    except Exception as e:
        warnings.warn(f"Exception {e} has been thrown during parsing")
        return None


def guess_mime_type(path: str) -> str:
    return mimetypes.guess_type(path)[0]


def get_guess_mime_type_udf():
    """Create the guess_mime_type UDF when needed."""
    return F.udf(guess_mime_type, T.StringType())


def get_parse_bytes_udf():
    """Create the parse_bytes UDF when needed."""
    return F.udf(parse_bytes, T.StringType())


def read_as_chunk(batch_iter: Iterator[pd.Series]) -> Iterator[pd.Series]:
    """
    Simple text chunking using sentence boundaries and character limits.
    Removes dependency on transformers and llama-index for Databricks compatibility.
    """

    def simple_sentence_split(
        text: str, max_chunk_size: int = 500, overlap: int = 10
    ) -> list[str]:
        """Split text into chunks based on sentence boundaries."""
        if not text or not text.strip():
            return []

        # Split on sentence boundaries (., !, ?)
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            # If adding this sentence would exceed max_chunk_size, start a new chunk
            if (
                current_chunk
                and len(current_chunk) + len(sentence) + 1 > max_chunk_size
            ):
                chunks.append(current_chunk.strip())
                # Start new chunk with overlap from previous chunk
                words = current_chunk.split()
                overlap_words = words[-overlap:] if len(words) > overlap else words
                current_chunk = " ".join(overlap_words) + " " + sentence
            else:
                current_chunk = (current_chunk + " " + sentence).strip()

        # Add the last chunk if it exists
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks if chunks else [""]

    def extract_and_split(doc_bytes: bytes) -> list[str]:
        """Extract text and split into chunks."""
        try:
            txt = parse_bytes(doc_bytes)
            if txt is None or not txt.strip():
                return []
            return simple_sentence_split(txt)
        except Exception as e:
            warnings.warn(f"Error processing document: {e}")
            return []

    for batch in batch_iter:
        yield batch.apply(extract_and_split)


def get_read_as_chunk_udf():
    """Create the read_as_chunk pandas UDF when needed."""
    return F.pandas_udf(read_as_chunk, "array<string>")

import mimetypes
import warnings
from io import BytesIO
import pandas as pd
from typing import Iterator

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
from llama_index.core import Document, set_global_tokenizer
from llama_index.core.node_parser import SentenceSplitter
from transformers import AutoTokenizer

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


@F.udf(T.StringType())
def guess_mime_type(path: str) -> str:
    return mimetypes.guess_type(path)[0]


@F.pandas_udf("array<string>")
def read_as_chunk(batch_iter: Iterator[pd.Series]) -> Iterator[pd.Series]:
    # set llama2 as tokenizer to match our model size (will stay below gte 1024 limit)
    set_global_tokenizer(
        AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    )
    # Sentence splitter from llama_index to split on sentences
    splitter = SentenceSplitter(chunk_size=500, chunk_overlap=10)

    def extract_and_split(b):
        txt = parse_bytes(b)
        if txt is None:
            return []
        nodes = splitter.get_nodes_from_documents([Document(text=txt)])
        return [n.text for n in nodes]

    for x in batch_iter:
        yield x.apply(extract_and_split)

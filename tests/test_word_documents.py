import unittest
import zipfile
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

from rag_demo.word_documents import (
    WordDocumentError,
    WordDocumentStore,
    build_word_chunks,
    extract_docx_blocks,
    normalize_word_filename,
)


def make_docx_bytes():
    document_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p>
      <w:pPr><w:pStyle w:val="Heading1"/></w:pPr>
      <w:r><w:t>專案說明</w:t></w:r>
    </w:p>
    <w:p><w:r><w:t>唯一測試碼是 AURORA-731，負責人是 Ryan。</w:t></w:r></w:p>
    <w:tbl>
      <w:tr>
        <w:tc><w:p><w:r><w:t>欄位</w:t></w:r></w:p></w:tc>
        <w:tc><w:p><w:r><w:t>內容</w:t></w:r></w:p></w:tc>
      </w:tr>
      <w:tr>
        <w:tc><w:p><w:r><w:t>預算</w:t></w:r></w:p></w:tc>
        <w:tc><w:p><w:r><w:t>42 萬元</w:t></w:r></w:p></w:tc>
      </w:tr>
    </w:tbl>
    <w:sectPr/>
  </w:body>
</w:document>
"""
    styles_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:style w:type="paragraph" w:styleId="Heading1">
    <w:name w:val="heading 1"/>
  </w:style>
</w:styles>
"""
    output = BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("word/document.xml", document_xml)
        archive.writestr("word/styles.xml", styles_xml)
    return output.getvalue()


class WordDocumentsTest(unittest.TestCase):
    def test_extracts_headings_paragraphs_and_tables(self):
        blocks = extract_docx_blocks(make_docx_bytes())

        self.assertEqual(blocks[0], {"kind": "heading", "text": "專案說明"})
        self.assertIn("AURORA-731", blocks[1]["text"])
        self.assertEqual(blocks[2]["kind"], "table")
        self.assertIn("預算 | 42 萬元", blocks[2]["text"])

        chunks = build_word_chunks(
            blocks,
            source_id="upload-0123456789abcdefabcd",
            filename="任意專案.docx",
            chunk_size=300,
            chunk_stride=180,
        )
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["source_id"], "upload-0123456789abcdefabcd")
        self.assertIn("任意專案.docx | 專案說明", chunks[0]["title"])
        self.assertIn("[表格 1]", chunks[0]["content"])

    def test_store_persists_and_deduplicates_an_indexed_document(self):
        with TemporaryDirectory() as directory:
            store = WordDocumentStore(
                root=Path(directory),
                embedding_model="fake-embedding",
                embed_chunks_fn=lambda chunks: [[1.0, float(index + 1)] for index, _ in enumerate(chunks)],
                now_fn=lambda: datetime(2026, 7, 21, 8, 30, tzinfo=timezone.utc),
            )

            first = store.import_docx("Ryan 專案.docx", make_docx_bytes())
            second = store.import_docx("同內容不同檔名.docx", make_docx_bytes())

            self.assertFalse(first.duplicate)
            self.assertTrue(second.duplicate)
            self.assertEqual(first.metadata["source_id"], second.metadata["source_id"])
            self.assertEqual(first.metadata["chunk_count"], len(first.chunks))
            self.assertTrue(store.original_path(first.metadata["source_id"]).is_file())
            self.assertEqual(store.list_documents()[0]["name"], "Ryan 專案.docx")

            reloaded = WordDocumentStore(
                root=Path(directory),
                embedding_model="fake-embedding",
                embed_chunks_fn=lambda chunks: [],
            ).load_indexed_documents()
            self.assertEqual(len(reloaded), 1)
            self.assertIn("AURORA-731", reloaded[0].chunks[0]["content"])
            self.assertEqual(len(reloaded[0].chunks), len(reloaded[0].embeddings))

    def test_rejects_legacy_doc_and_invalid_archives(self):
        with self.assertRaisesRegex(WordDocumentError, "另存為 .docx"):
            normalize_word_filename("old-format.doc")
        with self.assertRaisesRegex(WordDocumentError, "有效或可讀取"):
            extract_docx_blocks(b"not a zip file")


if __name__ == "__main__":
    unittest.main()

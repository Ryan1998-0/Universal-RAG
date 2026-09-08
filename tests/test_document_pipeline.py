import tempfile
import unittest
import zipfile
from io import BytesIO
from pathlib import Path

from rag_demo.document_pipeline import (
    DocumentPipelineError,
    DocumentStore,
    UNCATEGORIZED_FOLDER_ID,
    extract_document,
    normalize_document_filename,
)


def fake_embeddings(chunks):
    return [[1.0, float(index + 1), 0.5] for index, _ in enumerate(chunks)]


class FakePdfPage:
    def __init__(self, text):
        self.text = text

    def extract_text(self):
        return self.text


class FakePdfReader:
    is_encrypted = False

    def __init__(self, pages):
        self.pages = pages


def make_docx_with_image_bytes():
    document_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document
  xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
  xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
  xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
  xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing">
  <w:body>
    <w:p><w:r><w:t>先開啟付款資料。</w:t></w:r></w:p>
    <w:p>
      <w:r>
        <w:drawing>
          <wp:inline>
            <wp:docPr id="1" name="Picture 1" descr="付款日期輸入畫面"/>
            <a:graphic><a:graphicData><a:blip r:embed="rId5"/></a:graphicData></a:graphic>
          </wp:inline>
        </w:drawing>
      </w:r>
    </w:p>
    <w:sectPr/>
  </w:body>
</w:document>
"""
    relationships_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId5" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/image1.png"/>
</Relationships>
"""
    output = BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("word/document.xml", document_xml)
        archive.writestr("word/_rels/document.xml.rels", relationships_xml)
        archive.writestr("word/media/image1.png", b"fake-png")
    return output.getvalue()


class DocumentPipelineTest(unittest.TestCase):
    def test_supported_extensions_are_normalized_and_legacy_doc_is_rejected(self):
        name, extension, spec = normalize_document_filename("folder/Guide.JSON")

        self.assertEqual(name, "Guide.JSON")
        self.assertEqual(extension, ".json")
        self.assertEqual(spec["source_type"], "json")
        with self.assertRaisesRegex(DocumentPipelineError, "另存為 .docx"):
            normalize_document_filename("legacy.doc")

    def test_txt_import_is_persisted_unselected_and_deduplicated(self):
        with tempfile.TemporaryDirectory() as directory:
            store = DocumentStore(
                root=Path(directory),
                embedding_model="fake-embedding",
                embed_chunks_fn=fake_embeddings,
            )
            indexed = store.import_document(
                "policy.txt",
                "唯一識別碼 TXT-731，年度特休為十四日。".encode("utf-8"),
                content_type="text/plain",
            )
            duplicate = store.import_document(
                "renamed.txt",
                "唯一識別碼 TXT-731，年度特休為十四日。".encode("utf-8"),
            )
            reloaded = DocumentStore(
                root=Path(directory),
                embedding_model="fake-embedding",
                embed_chunks_fn=fake_embeddings,
            ).load_indexed_documents()

        self.assertFalse(indexed.duplicate)
        self.assertFalse(indexed.metadata["selected_by_default"])
        self.assertEqual(indexed.metadata["status"], "ready")
        self.assertEqual(indexed.metadata["source_type"], "text")
        self.assertIn("TXT-731", indexed.chunks[0]["content"])
        self.assertTrue(duplicate.duplicate)
        self.assertEqual(len(reloaded), 1)

    def test_identical_bytes_with_different_parsers_are_not_deduplicated(self):
        with tempfile.TemporaryDirectory() as directory:
            store = DocumentStore(
                root=Path(directory),
                embedding_model="fake-embedding",
                embed_chunks_fn=fake_embeddings,
            )
            text_document = store.import_document("same.txt", b"# Heading\nShared body")
            markdown_document = store.import_document("same.md", b"# Heading\nShared body")

        self.assertNotEqual(
            text_document.metadata["source_id"],
            markdown_document.metadata["source_id"],
        )
        self.assertEqual(text_document.metadata["source_type"], "text")
        self.assertEqual(markdown_document.metadata["source_type"], "markdown")

    def test_folders_persist_and_delete_moves_documents_to_uncategorized(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = DocumentStore(
                root=root,
                embedding_model="fake-embedding",
                embed_chunks_fn=fake_embeddings,
            )
            law_folder = store.create_folder("勞基法")
            case_folder = store.create_folder("法院案例")
            indexed = store.import_document(
                "article.txt",
                "勞動基準法第 16 條預告期間。".encode("utf-8"),
                folder_id=law_folder["id"],
            )

            self.assertEqual(indexed.metadata["folder_name"], "勞基法")
            self.assertEqual(store.list_folders()[1]["document_count"], 1)

            renamed = store.rename_folder(law_folder["id"], "勞動法規")
            moved = store.move_document(indexed.metadata["source_id"], case_folder["id"])
            deleted = store.delete_folder(case_folder["id"])

            reloaded = DocumentStore(
                root=root,
                embedding_model="fake-embedding",
                embed_chunks_fn=fake_embeddings,
            )
            documents = reloaded.list_documents()
            folders = reloaded.list_folders()

        self.assertEqual(renamed["name"], "勞動法規")
        self.assertEqual(moved["folder_name"], "法院案例")
        self.assertEqual(deleted["moved_document_count"], 1)
        self.assertEqual(documents[0]["folder_id"], UNCATEGORIZED_FOLDER_ID)
        self.assertEqual(documents[0]["folder_name"], "未分類")
        self.assertEqual([folder["name"] for folder in folders], ["未分類", "勞動法規"])

    def test_duplicate_upload_can_be_reclassified_without_reembedding(self):
        calls = []

        def track_embeddings(chunks):
            calls.append(len(chunks))
            return fake_embeddings(chunks)

        with tempfile.TemporaryDirectory() as directory:
            store = DocumentStore(
                root=Path(directory),
                embedding_model="fake-embedding",
                embed_chunks_fn=track_embeddings,
            )
            first_folder = store.create_folder("法條")
            second_folder = store.create_folder("案例")
            payload = "唯一代碼 FOLDER-200".encode("utf-8")
            store.import_document("same.txt", payload, folder_id=first_folder["id"])
            duplicate = store.import_document(
                "same.txt",
                payload,
                folder_id=second_folder["id"],
            )

        self.assertTrue(duplicate.duplicate)
        self.assertEqual(duplicate.metadata["folder_id"], second_folder["id"])
        self.assertEqual(calls, [1])

    def test_folder_names_are_unique_and_system_folder_is_protected(self):
        with tempfile.TemporaryDirectory() as directory:
            store = DocumentStore(root=Path(directory), embedding_model="fake-embedding")
            folder = store.create_folder("勞基法")
            with self.assertRaisesRegex(DocumentPipelineError, "相同名稱"):
                store.create_folder(" 勞基法 ")
            with self.assertRaisesRegex(DocumentPipelineError, "系統保留"):
                store.create_folder("未分類")
            with self.assertRaisesRegex(DocumentPipelineError, "不能重新命名或刪除"):
                store.delete_folder(UNCATEGORIZED_FOLDER_ID)
            with self.assertRaisesRegex(DocumentPipelineError, "找不到指定"):
                store.move_document("upload-0123456789abcdefabcd", "folder-000000000000")

        self.assertEqual(folder["document_count"], 0)

    def test_markdown_headings_become_separate_units(self):
        result = extract_document(
            "notes.md",
            "# Alpha\n代碼 MD-101\n## Beta\n第二節內容".encode("utf-8"),
            ".md",
        )

        self.assertEqual([unit["title"] for unit in result.units], ["Alpha", "Beta"])
        self.assertEqual(result.method, "markdown-headings")

    def test_json_is_validated_and_flattened_to_paths(self):
        result = extract_document(
            "record.json",
            b'{"customer":{"name":"Ryan","codes":["JSON-42"]}}',
            ".json",
        )

        self.assertIn('$.customer.name: "Ryan"', result.units[0]["content"])
        self.assertIn('$.customer.codes[0]: "JSON-42"', result.units[0]["content"])
        with self.assertRaisesRegex(DocumentPipelineError, "JSON 格式錯誤"):
            extract_document("broken.json", b'{"missing":', ".json")

    def test_image_uses_ocr_result_and_preserves_confidence(self):
        result = extract_document(
            "receipt.png",
            b"not-a-real-image-for-injected-ocr",
            ".png",
            ocr_fn=lambda _: {
                "text": "發票號碼 IMAGE-990，金額 1280 元",
                "confidence": 0.94,
                "engine": "test-vision",
                "languages": ["zh-Hant"],
            },
        )

        self.assertTrue(result.ocr_used)
        self.assertEqual(result.units[0]["ocr_confidence"], 0.94)
        self.assertIn("IMAGE-990", result.units[0]["content"])

    def test_docx_images_are_ocrd_and_described_by_vlm(self):
        result = extract_document(
            "付款手冊.docx",
            make_docx_with_image_bytes(),
            ".docx",
            ocr_fn=lambda _: {
                "text": "請輸入開始日期 2025/12/3",
                "confidence": 0.91,
                "engine": "test-ocr",
            },
            image_understanding_fn=lambda _: {
                "text": "終端機正在要求輸入付款清單的開始與結束日期。",
                "engine": "test-vlm",
                "model": "test-vision-model",
            },
        )

        self.assertEqual(result.method, "docx-xml+ocr+vlm")
        self.assertTrue(result.ocr_used)
        self.assertEqual(result.details["embedded_image_count"], 1)
        self.assertEqual(result.details["recognized_image_count"], 1)
        self.assertEqual(result.details["ocr_image_count"], 1)
        self.assertEqual(result.details["vlm_image_count"], 1)
        self.assertEqual(result.details["vision_models"], ["test-vision-model"])
        self.assertEqual(len(result.units), 2)
        self.assertIn("先開啟付款資料", result.units[0]["content"])
        self.assertIn("2025/12/3", result.units[1]["content"])
        self.assertIn("開始與結束日期", result.units[1]["content"])
        self.assertEqual(result.units[1]["ocr_confidence"], 0.91)

    def test_pdf_prefers_text_layer(self):
        result = extract_document(
            "guide.pdf",
            b"fake-pdf",
            ".pdf",
            pdf_reader_factory=lambda _: FakePdfReader(
                [FakePdfPage("PDF-200 is a unique verification code with enough text.")]
            ),
            pdf_page_renderer=lambda *_: self.fail("text PDF should not be rendered"),
        )

        self.assertFalse(result.ocr_used)
        self.assertEqual(result.method, "pdf-text")
        self.assertEqual(result.details["text_pages"], 1)

    def test_scanned_pdf_falls_back_to_page_ocr(self):
        def render_page(_pdf_path, _page_number, output_dir):
            image_path = output_dir / "page.png"
            image_path.write_bytes(b"fake-image")
            return image_path

        result = extract_document(
            "scan.pdf",
            b"fake-pdf",
            ".pdf",
            ocr_fn=lambda _: {
                "text": "掃描頁唯一代碼 SCAN-808",
                "confidence": 0.88,
                "engine": "test-vision",
            },
            pdf_reader_factory=lambda _: FakePdfReader([FakePdfPage("")]),
            pdf_page_renderer=render_page,
        )

        self.assertTrue(result.ocr_used)
        self.assertEqual(result.method, "pdf-ocr")
        self.assertEqual(result.details["ocr_pages"], 1)
        self.assertEqual(result.units[0]["extraction_method"], "test-vision")


if __name__ == "__main__":
    unittest.main()

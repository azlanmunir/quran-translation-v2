from __future__ import annotations

import sqlite3
import unittest

from quran_translate.db import init_db
from quran_translate.source_import import import_tanzil_xml
from quran_translate.validation import validate_source


class SourceImportTests(unittest.TestCase):
    def test_imported_tanzil_source_is_complete(self) -> None:
        conn = sqlite3.connect(":memory:")
        self.addCleanup(conn.close)
        conn.row_factory = sqlite3.Row
        init_db(conn)

        result = import_tanzil_xml(conn)
        self.assertEqual(result["surahs"], 114)
        self.assertEqual(result["ayahs"], 6236)
        self.assertEqual(validate_source(conn), [])


if __name__ == "__main__":
    unittest.main()

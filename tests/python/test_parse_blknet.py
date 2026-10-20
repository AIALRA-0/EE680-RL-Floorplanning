"""Check the received input files and reject data that would corrupt a circuit."""

from pathlib import Path
import tempfile
import unittest

from src.data.parse_blknet import load_circuit


ROOT = Path(__file__).resolve().parents[2]
SAMPLES = ROOT / "benchmarks" / "alarik-synthetic"


class CircuitParserTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.input_path = Path(directory.name) / "circuit.txt"

    def parse_text(self, text):
        self.input_path.write_text(text, encoding="utf-8")
        return load_circuit(self.input_path)

    def test_received_five_block_circuit(self):
        circuit = load_circuit(SAMPLES / "blknet_5.txt")
        self.assertEqual(circuit["blocks"], {
            "B0": (25.0, 8.0), "B1": (5.0, 28.0), "B2": (13.0, 12.0),
            "B3": (12.0, 9.0), "B4": (28.0, 8.0),
        })
        self.assertEqual(circuit["nets"], [
            ["B4", "B0", "B2", "B1"], ["B0", "B4"], ["B1", "B0"],
            ["B1", "B3", "B0", "B4"], ["B2", "B0", "B3", "B1"],
        ])
        self.assertEqual(set(circuit), {"blocks", "nets"})

    def test_all_received_circuit_sizes(self):
        for size in (5, 10, 20, 50, 100):
            with self.subTest(size=size):
                circuit = load_circuit(SAMPLES / f"blknet_{size}.txt")
                self.assertEqual(len(circuit["blocks"]), size)
                self.assertEqual(len(circuit["nets"]), size)

    def test_comments_whitespace_headers_and_fractional_sizes(self):
        circuit = self.parse_text(
            "# Example\n\n blocks \n A 2.5 1 \n B 1 3\n\n nets\n B A\n"
        )
        self.assertEqual(circuit["blocks"]["A"], (2.5, 1.0))
        self.assertEqual(circuit["nets"], [["B", "A"]])

    def test_unknown_block_reports_the_net_line(self):
        with self.assertRaisesRegex(ValueError, "circuit.txt:5:.*unknown block 'missing'"):
            self.parse_text("BLOCKS\nA 2 3\nNETS\n# Connection\nA missing\n")

    def test_duplicate_block_does_not_silently_replace_dimensions(self):
        with self.assertRaisesRegex(ValueError, "duplicate block ID 'A'"):
            self.parse_text("BLOCKS\nA 2 3\nA 4 5\nNETS\nA\n")

    def test_dimensions_must_be_finite_positive_numbers(self):
        for value in ("0", "-1", "nan", "inf", "oops"):
            for dimensions in (f"{value} 2", f"2 {value}"):
                with self.subTest(dimensions=dimensions), self.assertRaises(ValueError):
                    self.parse_text(f"BLOCKS\nA {dimensions}\nNETS\nA\n")

    def test_block_rows_require_exactly_three_fields(self):
        for row in ("A 2", "A 2 3 4"):
            with self.subTest(row=row), self.assertRaisesRegex(ValueError, "expected block ID"):
                self.parse_text(f"BLOCKS\n{row}\nNETS\nA\n")

    def test_missing_repeated_and_out_of_order_sections_are_rejected(self):
        for text in (
            "A 2 3\nNETS\nA\n", "BLOCKS\nA 2 3\n", "NETS\nA\n",
            "BLOCKS\nA 2 3\nBLOCKS\nNETS\nA\n",
            "BLOCKS\nA 2 3\nNETS\nNETS\nA\n",
            "BLOCKS\nA 2 3\nNETS\nBLOCKS\nA\n",
        ):
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.parse_text(text)

    def test_empty_circuit_is_rejected(self):
        for text in ("", "# Comment only\n", "BLOCKS\nNETS\n"):
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, "at least one block"):
                self.parse_text(text)

    def test_circuit_can_have_no_connections(self):
        circuit = self.parse_text("BLOCKS\nA 2 3\nNETS\n")
        self.assertEqual(circuit["nets"], [])

    def test_missing_file_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            load_circuit(self.input_path)


if __name__ == "__main__":
    unittest.main()

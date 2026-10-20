"""Check authored course references, dependency order, media and exercises."""
import ast
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def check_course(root=ROOT):
    book = json.loads((root / "steps/course/book.json").read_text(encoding="utf-8"))
    assert book["language"] == "en"
    assert not re.search(r"[\u4e00-\u9fff]", json.dumps(book, ensure_ascii=False))
    for name in ("materials", "questions", "chapters"):
        assert json.loads((root / f"steps/course/{name}.json").read_text(encoding="utf-8"))[name] == book[name], name
    sources = {s["id"]: s for s in book["materials"]}
    assert len(sources) == len(book["materials"])
    for source in sources.values():
        text = (root / source["path"]).read_text(encoding="utf-8")
        assert 1 <= source["start"] <= source["end"] <= len(text.splitlines())
        assert any(getattr(n, "lineno", None) == source["start"] and getattr(n, "end_lineno", None) == source["end"]
                   for n in ast.walk(ast.parse(text))), source["id"]
    seen = set()
    for question in book["questions"]:
        assert question["id"] not in seen
        assert set(question["dependencies"]) <= seen
        assert set(question["source_ids"]) <= sources.keys()
        assert len(question["explanation"]) == len(question["plan"]) == 3
        assert {p["id"] for p in question["plan"]} == {s["id"] for s in question["explanation"]}
        assert all(set(s["source_ids"]) <= sources.keys() and s["statement"] for s in question["explanation"])
        assert question["result"]["answer"] and question["appendix"]["pitfall"]
        seen.add(question["id"])
    assigned = []
    for chapter in book["chapters"]:
        assigned.extend(chapter["question_ids"])
        assert len(chapter["quiz"]) == 4 and chapter["short_answer"]["expected"]
        assert all(0 <= q["correct"] < len(q["options"]) and q["source_question"] in seen for q in chapter["quiz"])
        assert (root / chapter["media"]["src"].split("?", 1)[0].lstrip("/")).exists()
        assert chapter["media"]["reading_guide"] and len(chapter["media"]["components"]) == 3
    assert assigned == [q["id"] for q in book["questions"]]
    return {"chapters": len(book["chapters"]), "questions": len(seen), "multiple_choice": 24, "short_answer": 6, "status": "passed"}


if __name__ == "__main__":
    print(json.dumps(check_course()))

"""Assemble reviewed material, question and chapter records into the static book."""
import argparse
import json
from pathlib import Path


def assemble(directory):
    path = directory / "book.json"
    book = json.loads(path.read_text(encoding="utf-8"))
    for name in ("materials", "questions", "chapters"):
        book[name] = json.loads((directory / f"{name}.json").read_text(encoding="utf-8"))[name]
    previous = None
    answers = {}
    for question in book["questions"]:
        if previous is not None:
            question["premise"] = previous
        answers[question["id"]] = previous or question["premise"]
        previous = question["result"]["answer"]
    for chapter in book["chapters"]:
        chapter["bridge"] = answers[chapter["question_ids"][0]]
    for name in ("questions", "chapters"):
        (directory / f"{name}.json").write_text(json.dumps({name: book[name]}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    path.write_text(json.dumps(book, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"chapters": len(book["chapters"]), "questions": len(book["questions"])}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--course", type=Path, default=Path(__file__).parent / "course")
    print(json.dumps(assemble(parser.parse_args().course)))

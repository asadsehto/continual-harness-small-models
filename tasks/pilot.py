# The 20 pilot tasks: 15 train, 5 held-out. 10 coding, 10 pc.
# check(workdir, task) and reference(workdir, task) take the task dict too,
# so checkers can confirm that test files were not edited.

import os

from tasks.task import read, write, run_python, tests_pass, unchanged


# ---------------- coding (train) ----------------

def check_fix_fib(workdir, task):
    return unchanged(workdir, task, "test_fib.py") and tests_pass(workdir, "test_fib.py")


def ref_fix_fib(workdir, task):
    code = read(workdir, "fib.py").replace("for _ in range(n - 1):", "for _ in range(n):")
    write(workdir, "fib.py", code)


FIX_FIB = {
    "id": "fix_fib",
    "category": "coding",
    "split": "train",
    "instruction": "The tests in test_fib.py fail. Fix the bug in fib.py so all tests pass. Do not change test_fib.py.",
    "files": {
        "fib.py": "def fib(n):\n    a, b = 0, 1\n    for _ in range(n - 1):\n        a, b = b, a + b\n    return a\n",
        "test_fib.py": "import unittest\nfrom fib import fib\n\n\nclass TestFib(unittest.TestCase):\n    def test_values(self):\n        self.assertEqual(fib(0), 0)\n        self.assertEqual(fib(1), 1)\n        self.assertEqual(fib(2), 1)\n        self.assertEqual(fib(10), 55)\n\n\nif __name__ == '__main__':\n    unittest.main()\n",
    },
    "check": check_fix_fib,
    "reference": ref_fix_fib,
}


def check_palindrome(workdir, task):
    return unchanged(workdir, task, "test_text.py") and tests_pass(workdir, "test_text.py")


def ref_palindrome(workdir, task):
    write(workdir, "text.py", "def is_palindrome(s):\n    clean = ''.join(c.lower() for c in s if c.isalnum())\n    return clean == clean[::-1]\n")


PALINDROME = {
    "id": "palindrome",
    "category": "coding",
    "split": "train",
    "instruction": "Implement is_palindrome(s) in text.py. It must ignore case, spaces and punctuation. Make the tests in test_text.py pass without changing them.",
    "files": {
        "text.py": "def is_palindrome(s):\n    pass\n",
        "test_text.py": "import unittest\nfrom text import is_palindrome\n\n\nclass TestPal(unittest.TestCase):\n    def test_basic(self):\n        self.assertTrue(is_palindrome('racecar'))\n        self.assertFalse(is_palindrome('hello'))\n\n    def test_clean(self):\n        self.assertTrue(is_palindrome('A man, a plan, a canal: Panama'))\n        self.assertTrue(is_palindrome(''))\n\n\nif __name__ == '__main__':\n    unittest.main()\n",
    },
    "check": check_palindrome,
    "reference": ref_palindrome,
}


def check_import(workdir, task):
    result = run_python(workdir, ["main.py"])
    return result.returncode == 0 and result.stdout.strip() == "12"


def ref_import(workdir, task):
    write(workdir, "main.py", read(workdir, "main.py").replace("from mathutil import", "from mathutils import"))


FIX_IMPORT = {
    "id": "fix_import",
    "category": "coding",
    "split": "train",
    "instruction": "Running `python main.py` crashes. Fix it so it runs and prints the result. Do not change mathutils.py.",
    "files": {
        "mathutils.py": "def multiply(a, b):\n    return a * b\n",
        "main.py": "from mathutil import multiply\n\nprint(multiply(3, 4))\n",
    },
    "check": check_import,
    "reference": ref_import,
}


def check_word_count(workdir, task):
    write(workdir, "_check_input.txt", "one two  three\nfour\n\nfive six\n")
    result = run_python(workdir, ["count_words.py", "_check_input.txt"])
    os.remove(os.path.join(workdir, "_check_input.txt"))
    return result.returncode == 0 and result.stdout.strip() == "6"


def ref_word_count(workdir, task):
    write(workdir, "count_words.py", "import sys\n\nwith open(sys.argv[1]) as f:\n    print(len(f.read().split()))\n")


WORD_COUNT = {
    "id": "word_count",
    "category": "coding",
    "split": "train",
    "instruction": "Write a Python script count_words.py. `python count_words.py FILE` must print only the number of words in FILE (words are separated by any whitespace). Test it on notes.txt.",
    "files": {"notes.txt": "the quick brown fox\njumps over the lazy dog\n"},
    "check": check_word_count,
    "reference": ref_word_count,
}


def check_safe_divide(workdir, task):
    return unchanged(workdir, task, "test_calc.py") and tests_pass(workdir, "test_calc.py")


def ref_safe_divide(workdir, task):
    write(workdir, "calc.py", "def safe_divide(a, b):\n    if b == 0:\n        return None\n    return a / b\n")


SAFE_DIVIDE = {
    "id": "safe_divide",
    "category": "coding",
    "split": "train",
    "instruction": "safe_divide in calc.py crashes when b is 0. Make it return None in that case. All tests in test_calc.py must pass; do not change the tests.",
    "files": {
        "calc.py": "def safe_divide(a, b):\n    return a / b\n",
        "test_calc.py": "import unittest\nfrom calc import safe_divide\n\n\nclass TestDiv(unittest.TestCase):\n    def test_normal(self):\n        self.assertEqual(safe_divide(10, 4), 2.5)\n\n    def test_zero(self):\n        self.assertIsNone(safe_divide(1, 0))\n\n\nif __name__ == '__main__':\n    unittest.main()\n",
    },
    "check": check_safe_divide,
    "reference": ref_safe_divide,
}


def check_rename_function(workdir, task):
    shop = read(workdir, "shop.py") or ""
    if "def calculate_total(" not in shop or "def calc(" in shop:
        return False
    result = run_python(workdir, ["main.py"])
    return result.returncode == 0 and result.stdout.strip() == "35"


def ref_rename_function(workdir, task):
    write(workdir, "shop.py", read(workdir, "shop.py").replace("def calc(", "def calculate_total("))
    write(workdir, "main.py", read(workdir, "main.py").replace("calc", "calculate_total"))


RENAME_FUNCTION = {
    "id": "rename_function",
    "category": "coding",
    "split": "train",
    "instruction": "Rename the function calc in shop.py to calculate_total, and update every place that uses it. `python main.py` must still work.",
    "files": {
        "shop.py": "def calc(prices):\n    return sum(prices)\n",
        "main.py": "from shop import calc\n\nprint(calc([10, 20, 5]))\n",
    },
    "check": check_rename_function,
    "reference": ref_rename_function,
}


def check_json_port(workdir, task):
    import json
    text = read(workdir, "config.json")
    if text is None:
        return False
    try:
        data = json.loads(text)
    except ValueError:
        return False
    return data == {"host": "localhost", "port": 8080, "debug": False, "name": "app"}


def ref_json_port(workdir, task):
    write(workdir, "config.json", read(workdir, "config.json").replace("5000", "8080"))


JSON_PORT = {
    "id": "json_port",
    "category": "coding",
    "split": "train",
    "instruction": "In config.json, change the port to 8080 (as a number). Keep every other setting the same and keep the file valid JSON.",
    "files": {"config.json": '{\n  "host": "localhost",\n  "port": 5000,\n  "debug": false,\n  "name": "app"\n}\n'},
    "check": check_json_port,
    "reference": ref_json_port,
}


# ---------------- pc (train) ----------------

def check_organize(workdir, task):
    expected = {
        "txt/a.txt", "txt/b.txt", "csv/data.csv", "csv/more.csv",
    }
    found = set()
    for root, dirs, names in os.walk(workdir):
        for name in names:
            found.add(os.path.relpath(os.path.join(root, name), workdir))
    return found == expected


def ref_organize(workdir, task):
    for name in ["a.txt", "b.txt", "data.csv", "more.csv"]:
        ext = name.split(".")[1]
        os.makedirs(os.path.join(workdir, ext), exist_ok=True)
        os.rename(os.path.join(workdir, name), os.path.join(workdir, ext, name))


ORGANIZE = {
    "id": "organize",
    "category": "pc",
    "split": "train",
    "instruction": "Move every .txt file into a folder named txt and every .csv file into a folder named csv. Leave no other files behind.",
    "files": {"a.txt": "a\n", "b.txt": "b\n", "data.csv": "x,y\n1,2\n", "more.csv": "x,y\n3,4\n"},
    "check": check_organize,
    "reference": ref_organize,
}


def check_count_lines(workdir, task):
    answer = read(workdir, "answer.txt")
    return answer is not None and answer.strip() == "7"


def ref_count_lines(workdir, task):
    write(workdir, "answer.txt", "7\n")


COUNT_LINES = {
    "id": "count_lines",
    "category": "pc",
    "split": "train",
    "instruction": "Count the number of lines in server.log and write only that number into answer.txt.",
    "files": {"server.log": "start\nrequest 1\nrequest 2\nwarning disk\nrequest 3\nrequest 4\nstop\n"},
    "check": check_count_lines,
    "reference": ref_count_lines,
}


def check_csv_sum(workdir, task):
    answer = read(workdir, "total.txt")
    if answer is None:
        return False
    try:
        return abs(float(answer.strip()) - 351.25) < 0.001
    except ValueError:
        return False


def ref_csv_sum(workdir, task):
    write(workdir, "total.txt", "351.25\n")


CSV_SUM = {
    "id": "csv_sum",
    "category": "pc",
    "split": "train",
    "instruction": "sales.csv has a column named amount. Add up all values in that column and write only the total into total.txt.",
    "files": {"sales.csv": "date,item,amount\n2026-01-01,pen,12.50\n2026-01-02,book,99.00\n2026-01-03,bag,200.75\n2026-01-04,cup,39.00\n"},
    "check": check_csv_sum,
    "reference": ref_csv_sum,
}


ERROR_LINES = "2026-11-01 ERROR db timeout\n2026-11-01 ERROR disk full\n2026-11-02 ERROR auth failed\n"


def check_find_errors(workdir, task):
    errors = read(workdir, "errors.log")
    return errors is not None and errors.strip() == ERROR_LINES.strip() and unchanged(workdir, task, "app.log")


def ref_find_errors(workdir, task):
    write(workdir, "errors.log", ERROR_LINES)


FIND_ERRORS = {
    "id": "find_errors",
    "category": "pc",
    "split": "train",
    "instruction": "Copy every line of app.log that contains the word ERROR into a new file errors.log, in the same order. Do not change app.log.",
    "files": {"app.log": "2026-11-01 INFO start\n2026-11-01 ERROR db timeout\n2026-11-01 INFO retry\n2026-11-01 ERROR disk full\n2026-11-02 INFO ok\n2026-11-02 ERROR auth failed\n"},
    "check": check_find_errors,
    "reference": ref_find_errors,
}


def check_rename_photos(workdir, task):
    names = set(os.listdir(workdir))
    return names == {"photo_001.jpg", "photo_002.jpg", "photo_003.jpg"} and read(workdir, "photo_002.jpg") == "image two\n"


def ref_rename_photos(workdir, task):
    for i in range(1, 4):
        os.rename(os.path.join(workdir, "IMG_" + str(i) + ".JPG"), os.path.join(workdir, "photo_00" + str(i) + ".jpg"))


RENAME_PHOTOS = {
    "id": "rename_photos",
    "category": "pc",
    "split": "train",
    "instruction": "Rename IMG_1.JPG, IMG_2.JPG and IMG_3.JPG to photo_001.jpg, photo_002.jpg and photo_003.jpg (same number, lowercase extension).",
    "files": {"IMG_1.JPG": "image one\n", "IMG_2.JPG": "image two\n", "IMG_3.JPG": "image three\n"},
    "check": check_rename_photos,
    "reference": ref_rename_photos,
}


def check_delete_tmp(workdir, task):
    names = set()
    for root, dirs, files in os.walk(workdir):
        for name in files:
            names.add(os.path.relpath(os.path.join(root, name), workdir))
    return names == {"keep.txt", "notes/plan.md"}


def ref_delete_tmp(workdir, task):
    os.remove(os.path.join(workdir, "cache.tmp"))
    os.remove(os.path.join(workdir, "notes/draft.tmp"))


DELETE_TMP = {
    "id": "delete_tmp",
    "category": "pc",
    "split": "train",
    "instruction": "Delete every .tmp file in this folder and its subfolders. Keep all other files.",
    "files": {"keep.txt": "keep\n", "cache.tmp": "x\n", "notes/plan.md": "# plan\n", "notes/draft.tmp": "y\n"},
    "check": check_delete_tmp,
    "reference": ref_delete_tmp,
}


def check_merge(workdir, task):
    return read(workdir, "full.txt") == "Part one.\nPart two.\nPart three.\n"


def ref_merge(workdir, task):
    text = read(workdir, "part1.txt") + read(workdir, "part2.txt") + read(workdir, "part3.txt")
    write(workdir, "full.txt", text)


MERGE = {
    "id": "merge_parts",
    "category": "pc",
    "split": "train",
    "instruction": "Join part1.txt, part2.txt and part3.txt (in that order) into one new file full.txt.",
    "files": {"part3.txt": "Part three.\n", "part1.txt": "Part one.\n", "part2.txt": "Part two.\n"},
    "check": check_merge,
    "reference": ref_merge,
}


def check_scaffold(workdir, task):
    readme = read(workdir, "README.md")
    if readme is None or readme.splitlines()[0:1] != ["# weather-app"]:
        return False
    if not os.path.isdir(os.path.join(workdir, "src")) or not os.path.isdir(os.path.join(workdir, "tests")):
        return False
    return read(workdir, "src/__init__.py") is not None


def ref_scaffold(workdir, task):
    write(workdir, "README.md", "# weather-app\n")
    write(workdir, "src/__init__.py", "")
    os.makedirs(os.path.join(workdir, "tests"), exist_ok=True)


SCAFFOLD = {
    "id": "scaffold",
    "category": "pc",
    "split": "train",
    "instruction": "Create a project layout: a folder src containing an empty __init__.py, an empty folder tests, and a README.md whose first line is exactly: # weather-app",
    "files": {},
    "check": check_scaffold,
    "reference": ref_scaffold,
}


# ---------------- held-out (only evaluated, never learned from) ----------------

def check_reverse_words(workdir, task):
    return unchanged(workdir, task, "test_words.py") and tests_pass(workdir, "test_words.py")


def ref_reverse_words(workdir, task):
    write(workdir, "words.py", "def reverse_words(s):\n    return ' '.join(s.split()[::-1])\n")


REVERSE_WORDS = {
    "id": "reverse_words",
    "category": "coding",
    "split": "heldout",
    "instruction": "Implement reverse_words(s) in words.py so the tests in test_words.py pass. Do not change the tests.",
    "files": {
        "words.py": "def reverse_words(s):\n    pass\n",
        "test_words.py": "import unittest\nfrom words import reverse_words\n\n\nclass TestWords(unittest.TestCase):\n    def test_it(self):\n        self.assertEqual(reverse_words('hello big world'), 'world big hello')\n        self.assertEqual(reverse_words('  one   two '), 'two one')\n        self.assertEqual(reverse_words(''), '')\n\n\nif __name__ == '__main__':\n    unittest.main()\n",
    },
    "check": check_reverse_words,
    "reference": ref_reverse_words,
}


def check_average(workdir, task):
    return unchanged(workdir, task, "test_stats.py") and tests_pass(workdir, "test_stats.py")


def ref_average(workdir, task):
    write(workdir, "stats.py", "def average(nums):\n    if len(nums) == 0:\n        return 0\n    return sum(nums) / len(nums)\n")


AVERAGE = {
    "id": "average_empty",
    "category": "coding",
    "split": "heldout",
    "instruction": "The tests in test_stats.py fail. Fix stats.py so they pass. Do not change the tests.",
    "files": {
        "stats.py": "def average(nums):\n    return sum(nums) / len(nums)\n",
        "test_stats.py": "import unittest\nfrom stats import average\n\n\nclass TestAvg(unittest.TestCase):\n    def test_values(self):\n        self.assertEqual(average([2, 4, 6]), 4)\n\n    def test_empty(self):\n        self.assertEqual(average([]), 0)\n\n\nif __name__ == '__main__':\n    unittest.main()\n",
    },
    "check": check_average,
    "reference": ref_average,
}


def check_failed_rows(workdir, task):
    answer = read(workdir, "failed.txt")
    return answer is not None and answer.strip() == "3"


def ref_failed_rows(workdir, task):
    write(workdir, "failed.txt", "3\n")


FAILED_ROWS = {
    "id": "failed_rows",
    "category": "pc",
    "split": "heldout",
    "instruction": "jobs.csv has a status column. Count how many rows have status equal to failed and write only that number into failed.txt.",
    "files": {"jobs.csv": "id,status\n1,ok\n2,failed\n3,ok\n4,failed\n5,running\n6,failed\n7,ok\n"},
    "check": check_failed_rows,
    "reference": ref_failed_rows,
}


def check_largest(workdir, task):
    answer = read(workdir, "largest.txt")
    return answer is not None and answer.strip() == "video.bin"


def ref_largest(workdir, task):
    write(workdir, "largest.txt", "video.bin\n")


LARGEST = {
    "id": "largest_file",
    "category": "pc",
    "split": "heldout",
    "instruction": "Find the largest file (by size in bytes) among small.txt, video.bin, report.pdf and notes.md. Write only its file name into largest.txt.",
    "files": {
        "small.txt": "hi\n",
        "video.bin": "0123456789" * 50,
        "report.pdf": "0123456789" * 20,
        "notes.md": "0123456789" * 5,
    },
    "check": check_largest,
    "reference": ref_largest,
}


def check_spelling(workdir, task):
    for name in ["a.md", "docs/b.md"]:
        text = read(workdir, name) or ""
        if "colour" in text or "color" not in text:
            return False
    return unchanged(workdir, task, "keep.txt")


def ref_spelling(workdir, task):
    for name in ["a.md", "docs/b.md"]:
        write(workdir, name, read(workdir, name).replace("colour", "color"))


SPELLING = {
    "id": "us_spelling",
    "category": "pc",
    "split": "heldout",
    "instruction": "Replace every 'colour' with 'color' in all .md files in this folder and its subfolders. Do not change other file types.",
    "files": {
        "a.md": "Pick a colour. The colour is red.\n",
        "docs/b.md": "Background colour: blue\n",
        "keep.txt": "This colour stays.\n",
    },
    "check": check_spelling,
    "reference": ref_spelling,
}


ALL_TASKS = [
    FIX_FIB, PALINDROME, FIX_IMPORT, WORD_COUNT, SAFE_DIVIDE, RENAME_FUNCTION, JSON_PORT,
    ORGANIZE, COUNT_LINES, CSV_SUM, FIND_ERRORS, RENAME_PHOTOS, DELETE_TMP, MERGE, SCAFFOLD,
    REVERSE_WORDS, AVERAGE, FAILED_ROWS, LARGEST, SPELLING,
]

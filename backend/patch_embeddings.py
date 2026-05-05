import re
from pathlib import Path

for file_path in ["tests/test_manager_face_enrollment_api.py", "tests/test_manager_api.py"]:
    path = Path(file_path)
    if not path.exists():
        continue
    content = path.read_text(encoding="utf-8")

    # Replace specific returns
    content = content.replace("return [[0.1, 0.2, 0.3]]", "return [[0.1] * 512]")
    content = content.replace("return [[0.5, 0.2, 0.3]]", "return [[0.5] * 512]")
    content = content.replace("return [[0.9, 0.2, 0.3]]", "return [[0.9] * 512]")

    # Replace list comprehensions or dynamic returns
    content = re.sub(
        r'return \[\[float\(ord\(suffix\[-1\]\)\) / 100\.0, 0\.2, 0\.3\]\]',
        r'return [[float(ord(suffix[-1])) / 100.0] * 512]',
        content
    )
    content = re.sub(
        r'return \[\[0\.11 \+ \(suffix \* 0\.1\), 0\.2 \+ \(suffix % 3\) \* 0\.2, 0\.3\]\]',
        r'return [[0.11 + (suffix * 0.1)] * 512]',
        content
    )
    content = re.sub(
        r'return \[\[0\.1, 0\.2, 0\.3\], \[0\.4, 0\.5, 0\.6\]\]',
        r'return [[0.1] * 512, [0.4] * 512]',
        content
    )

    path.write_text(content, encoding="utf-8")


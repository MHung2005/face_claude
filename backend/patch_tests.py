import re

with open('tests/test_manager_face_enrollment_api.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = re.sub(
    r'class FakeFaceIndexService:\s+def refresh\(self\):\s+raise AssertionError\("refresh should not be called on failure"\)',
    'class FakeFaceIndexService:\n        def refresh(self):\n            raise AssertionError("refresh should not be called on failure")\n        def upsert(self, **kwargs):\n            raise AssertionError("upsert should not be called on failure")\n        def delete_employee(self, employee_id):\n            raise AssertionError("delete_employee should not be called on failure")',
    content
)

content = re.sub(
    r'class FakeFaceIndexService:\s+def refresh\(self\):\s+refresh_calls\["count"\] \+= 1\s+def delete_employee\(self, employee_id\):\s+pass',
    'class FakeFaceIndexService:\n        def refresh(self):\n            refresh_calls["count"] += 1\n        def upsert(self, **kwargs):\n            refresh_calls["count"] += 1\n        def delete_employee(self, employee_id):\n            refresh_calls["count"] += 1',
    content
)

content = re.sub(
    r'class FakeFaceIndexService:\s+def refresh\(self\):\s+refresh_calls\["count"\] \+= 1',
    'class FakeFaceIndexService:\n        def refresh(self):\n            refresh_calls["count"] += 1\n        def upsert(self, **kwargs):\n            refresh_calls["count"] += 1\n        def delete_employee(self, employee_id):\n            refresh_calls["count"] += 1',
    content
)

content = re.sub(
    r'class FakeFaceIndexService:\s+def refresh\(self\):\s+return None',
    'class FakeFaceIndexService:\n        def refresh(self):\n            return None\n        def upsert(self, **kwargs):\n            return None\n        def delete_employee(self, employee_id):\n            return None',
    content
)

with open('tests/test_manager_face_enrollment_api.py', 'w', encoding='utf-8') as f:
    f.write(content)

# And for test_manager_api.py
with open('tests/test_manager_api.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = re.sub(
    r'class FakeFaceIndexService:\s+def refresh\(self\):\s+refresh_calls\["count"\] \+= 1\s+def delete_employee\(self, employee_id\):\s+pass',
    'class FakeFaceIndexService:\n        def refresh(self):\n            refresh_calls["count"] += 1\n        def upsert(self, **kwargs):\n            refresh_calls["count"] += 1\n        def delete_employee(self, employee_id):\n            refresh_calls["count"] += 1',
    content
)

with open('tests/test_manager_api.py', 'w', encoding='utf-8') as f:
    f.write(content)

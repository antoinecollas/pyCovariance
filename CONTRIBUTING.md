For a file named `file.py`.
1. Create a corresponding test file: `tests/test_file.py`.
2. Run these tests with `nose2 tests.test_file -v --with-coverage`.
3. Run `coverage combine && coverage html`.
4. Check the coverage of `file.py` by opening `htmlcov/index.html`.
5. Check style of code by running `flake8 file.py tests/test_file.py`.

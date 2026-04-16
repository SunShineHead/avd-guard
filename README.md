📘 README.md — Pyton Package In Conda

A fully self‑healing, auto-fixing, auto‑versioned, auto‑releasing MLOps system.

 

🚀 Overview

OASIS is a fully autonomous Machine Learning + DevOps hybrid pipeline featuring:

Real‑dataset LightGBM training

Versioned model saving

Semantic versioning

Full CLI toolkit ( oasis train ,  oasis version ,  oasis auto‑fix , etc.)

Automatic changelog generation

Automatic GitHub Releases

CI Retry + Auto‑Merge system

PR‑based self‑healing

Auto‑close failing PRs

Nightly auto‑fix pipelines

Auto‑formatting, linting, diagnostics, and repository cleanup

OASIS maintains itself — heals its own repo, fixes CI failures, formats code, retries CI, publishes releases, updates changelogs, and more.

 

📁 Project Structure

 
OASIS/
│
├── data/
│   └── dataset.csv
│
├── models/
│   ├── trained_model.pkl
│   ├── version.txt
│   └── history.log
│
├── src/
│   ├── train_pipeline.py
│   ├── model_loader.py
│   └── oasis/
│       └── cli.py
│
├── tests/
│   └── test_lgb_model.py
│
└── .github/workflows/
    ├── ci.yml
    ├── oasis-auto-fix.yml
    ├── oasis-auto-fix-pr.yml
    ├── oasis-auto-fix-nightly.yml
    ├── oasis-auto-merge.yml
    ├── oasis-auto-close.yml
    └── oasis-ci-retry.yml
 

 

🧠 Training Pipeline

Training uses:

 
src/train_pipeline.py
 

Pipeline includes:

Loading real dataset

Splitting training/test

Training LightGBM

Saving model + metadata

Recording semantic version

Appending version history

Train manually:

 
oasis train
 

 

🧪 Testing

Tests validate:

Model load

Feature alignment

Prediction behavior

Deterministic output

Run manually:

 
pytest -v


pytest documentation

Logo
Search
Get Started
How-to guides
Reference guides
Explanation
Examples and customization tricks
About the project

Changelog
Contributing
Backwards Compatibility Policy
History
Python version support
Sponsor
pytest for enterprise
License
Contact channels
Useful links

pytest @ PyPI
pytest @ GitHub
Issue Tracker
PDF Documentation
Get Started
Install pytest
Run the following command in your command line:

pip install -U pytest
Check that you installed the correct version:

$ pytest --version
pytest 9.0.2
Create your first test
Create a new file called test_sample.py, containing a function, and a test:

# content of test_sample.py
def func(x):
    return x + 1


def test_answer():
    assert func(3) == 5
The test

$ pytest
=========================== test session starts ============================
platform linux -- Python 3.x.y, pytest-9.x.y, pluggy-1.x.y
rootdir: /home/sweet/project
collected 1 item

test_sample.py F                                                     [100%]

================================= FAILURES =================================
_______________________________ test_answer ________________________________

    def test_answer():
>       assert func(3) == 5
E       assert 4 == 5
E        +  where 4 = func(3)

test_sample.py:6: AssertionError
========================= short test summary info ==========================
FAILED test_sample.py::test_answer - assert 4 == 5
============================ 1 failed in 0.12s =============================
The [100%] refers to the overall progress of running all test cases. After it finishes, pytest then shows a failure report because func(3) does not return 5.

Note

You can use the assert statement to verify test expectations. pytest’s Advanced assertion introspection will intelligently report intermediate values of the assert expression so you can avoid the many names of JUnit legacy methods.

Run multiple tests
pytest will run all files of the form test_*.py or *_test.py in the current directory and its subdirectories. More generally, it follows standard test discovery rules.

Assert that a certain exception is raised
Use the raises helper to assert that some code raises an exception:

# content of test_sysexit.py
import pytest


def f():
    raise SystemExit(1)


def test_mytest():
    with pytest.raises(SystemExit):
        f()
Execute the test function with “quiet” reporting mode:

$ pytest -q test_sysexit.py
.                                                                    [100%]
1 passed in 0.12s
Note

The -q/--quiet flag keeps the output brief in this and following examples.

See Assertions about approximate equality for specifying more details about the expected exception.

Group multiple tests in a class
Once you develop multiple tests, you may want to group them into a class. pytest makes it easy to create a class containing more than one test:

# content of test_class.py
class TestClass:
    def test_one(self):
        x = "this"
        assert "h" in x

    def test_two(self):
        x = "hello"
        assert hasattr(x, "check")
pytest discovers all tests following its Conventions for Python test discovery, so it finds both test_ prefixed functions. There is no need to subclass anything, but make sure to prefix your class with Test otherwise the class will be skipped. We can simply run the module by passing its filename:

$ pytest -q test_class.py
.F                                                                   [100%]
================================= FAILURES =================================
____________________________ TestClass.test_two ____________________________

self = <test_class.TestClass object at 0xdeadbeef0001>

    def test_two(self):
        x = "hello"
>       assert hasattr(x, "check")
E       AssertionError: assert False
E        +  where False = hasattr('hello', 'check')

test_class.py:8: AssertionError
========================= short test summary info ==========================
FAILED test_class.py::TestClass::test_two - AssertionError: assert False
1 failed, 1 passed in 0.12s
The first test passed and the second failed. You can easily see the intermediate values in the assertion to help you understand the reason for the failure.

Grouping tests in classes can be beneficial for the following reasons:

Test organization

Sharing fixtures for tests only in that particular class

Applying marks at the class level and having them implicitly apply to all tests

Something to be aware of when grouping tests inside classes is that each test has a unique instance of the class. Having each test share the same class instance would be very detrimental to test isolation and would promote poor test practices. This is outlined below:

# content of test_class_demo.py
class TestClassDemoInstance:
    value = 0

    def test_one(self):
        self.value = 1
        assert self.value == 1

    def test_two(self):
        assert self.value == 1
$ pytest -k TestClassDemoInstance -q
.F                                                                   [100%]
================================= FAILURES =================================
______________________ TestClassDemoInstance.test_two ______________________

self = <test_class_demo.TestClassDemoInstance object at 0xdeadbeef0002>

    def test_two(self):
>       assert self.value == 1
E       assert 0 == 1
E        +  where 0 = <test_class_demo.TestClassDemoInstance object at 0xdeadbeef0002>.value

test_class_demo.py:9: AssertionError
========================= short test summary info ==========================
FAILED test_class_demo.py::TestClassDemoInstance::test_two - assert 0 == 1
1 failed, 1 passed in 0.12s
Note that attributes added at class level are class attributes, so they will be shared between tests.

Compare floating-point values with pytest.approx
pytest also provides a number of utilities to make writing tests easier. For example, you can use pytest.approx() to compare floating-point values that may have small rounding errors:

# content of test_approx.py
import pytest


def test_sum():
    assert (0.1 + 0.2) == pytest.approx(0.3)
This avoids the need for manual tolerance checks or using math.isclose and works with scalars, lists, and NumPy arrays.

Request a unique temporary directory for functional tests
pytest provides Builtin fixtures/function arguments to request arbitrary resources, like a unique temporary directory:

# content of test_tmp_path.py
def test_needsfiles(tmp_path):
    print(tmp_path)
    assert 0
List the name tmp_path in the test function signature and pytest will lookup and call a fixture factory to create the resource before performing the test function call. Before the test runs, pytest creates a unique-per-test-invocation temporary directory:

$ pytest -q test_tmp_path.py
F                                                                    [100%]
================================= FAILURES =================================
_____________________________ test_needsfiles ______________________________

tmp_path = PosixPath('PYTEST_TMPDIR/test_needsfiles0')

    def test_needsfiles(tmp_path):
        print(tmp_path)
>       assert 0
E       assert 0

test_tmp_path.py:3: AssertionError
--------------------------- Captured stdout call ---------------------------
PYTEST_TMPDIR/test_needsfiles0
========================= short test summary info ==========================
FAILED test_tmp_path.py::test_needsfiles - assert 0
1 failed in 0.12s
More info on temporary directory handling is available at Temporary directories and files.

Find out what kind of builtin pytest fixtures exist with the command:

pytest --fixtures   # shows builtin and custom fixtures
Note that this command omits fixtures with leading _ unless the -v option is added.

Continue reading
Check out additional pytest resources to help you customize tests for your unique workflow:

“How to invoke pytest” for command line invocation examples

“How to use pytest with an existing test suite” for working with preexisting tests

“How to mark test functions with attributes” for information on the pytest.mark mechanism

“Fixtures reference” for providing a functional baseline to your tests

“Writing plugins” for managing and writing plugins

“Good Integration Practices” for virtualenv and test layouts
 

 

⚙️ GitHub Actions Overview

OASIS includes 7 fully autonomous workflows:

✔  ci.yml 

Standard train + test workflow.

✔  oasis-auto-fix.yml 

Self-heals repository on command.

✔  oasis-auto-fix-pr.yml 

Creates auto-fix PRs instead of pushing changes.

✔  oasis-auto-fix-nightly.yml 

Runs nightly repository healing at 2AM UTC.

✔  oasis-auto-merge.yml 

Auto-merges approved auto-fix PRs only when CI is green.

✔  oasis-auto-close.yml 

Auto-closes persistent failing PRs after 3 CI failures.

✔  oasis-ci-retry.yml 

Retries CI up to 3 times before merging or closing.

Combined, these workflows create a self-maintaining MLOps ecosystem.

 

🧵 OASIS CLI Commands

Your CLI includes:

🔧 Training & Model Management

 
oasis train
oasis evaluate <dataset.csv>
oasis predict <input.csv>
 

🔍 Model Metadata

 
oasis version
oasis version --json
 

Metadata includes:

Semantic version

Timestamp

Feature list

Model size

File path

🧾 Version History & Releases

 
oasis bump-version --level patch|minor|major
oasis history
oasis changelog
oasis release
 

Release automatically:

Tags Git

Generates changelog

Uploads model to GitHub Releases

🛠 Auto‑Fix & Formatting

 
oasis auto-fix
oasis auto-fix-strict
oasis format
oasis clean
 

🩺 Diagnostics

 
oasis doctor
oasis doctor --json
oasis doctor --fix
oasis doctor --fix --commit --push
 

Doctor checks:

Python syntax

YAML health

GPU availability

Missing dependencies

Model file integrity

Git status

Auto-healing

 

🤖 Self‑Healing DevOps Explained

OASIS includes autonomous maintenance loops:

1️⃣ Failure → Auto-Fix PR

A CI failure triggers a repair branch & PR.

2️⃣ Auto‑Retry CI

OASIS retries CI up to 3 times.

3️⃣ Auto‑Comment Failure Reasons

Explains why CI failed directly on PR.

4️⃣ Auto‑Merge

If CI passes + PR is approved → merge.

5️⃣ Auto‑Close

If CI fails 3 times → PR closed with explanation.

6️⃣ Nightly Repair

Nightly self-healing runs regardless of CI.

 

🚀 Release Automation

Release with:

 
oasis release
 

This:

Reads semantic version

Creates Git tag

Generates changelog

Uploads model

Publishes GitHub Release

Optional:

 
oasis release --no-confirm
oasis release --notes "Custom message"
 

 

🧹 Cleanup & Formatting

Run:

 
oasis clean
oasis format
 

Removes:

Caches

Build files

Logs

Model artifacts (optional)

And formats code using:

Black

isort

docformatter

 

📦 Installation

Editable mode installation:

 
pip install -e .
 

 

🛟 Support

If you need enhancements, improvements, or more automation, extend the CLI or GitHub workflows.

 

🎉 Final Note

This README documents your complete autonomous ML + DevOps pipeline.
Your OASIS system is now capable of:

Training

Testing

Healing

Formatting

Releasing

Versioning

Closing

Commenting

Auto-merging

Nightly cleaning

all without human intervention.
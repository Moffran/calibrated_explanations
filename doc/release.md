Releasing calibrated-explanations on PyPI
=========================================

This is a guide on how to release calibrated-explanations on PyPI.


## Pre-requisites (once)

First make sure to understand the contents of [official Python packaging guide].

[official Python packaging guide]: https://packaging.python.org/en/latest/tutorials/packaging-projects/

Make sure you have the build and twine Python packages installed.
Use pip or your package manager of choice to install them.
Make sure you have a test.pypi.org and pypi.org accounts.
Set-up your .pypirc as instructed when creating a token:

```ini
[testpypi]
  username = __token__
  password = pypi-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

## Release steps (every time)

1. Bump the version number on pyproject.toml.

2. In the project root, clear up your `dist/` directory.

	```bash
	$ rm -r dist/
	```

3. Build a `dist/` of the latest version.

	```bash
	python -m build
	```

4. Upload your package using twine.

	```bash
	twine upload --repository testpypi dist/*
	```

5. Check that the package page was rendered correctly
   using the link reported by twine, in the form:

	https://test.pypi.org/project/calibrated-explanations/

6. Test installation of the package in a virtual environment:

	$ python -m venv venv
	$ source venv/bin/activate
	(venv) $ python3 -m pip install -r requirements.txt
	(venv) $ python3 -m pip install ipython  # TODO: move to requirements!
	(venv) $ python3 -m pip install -i https://test.pypi.org/simple/ --no-deps calibrated-explanations
	(venv) $ python3
	>>> import ce
	>>> ...

7. Upload your package to the real PyPI using twine:

	```bash
	twine upload pypi dist/*
	```

8. check that the project page was rendered correctly in:

	https://test.pypi.org/project/calibrated-explanations/

9. test installation

	$ pip install calibrated-explanations
	>>> import ce

This could be automated through CI,
but would only be worth it if we are releasing very often.
So for now, we just follow the guide.

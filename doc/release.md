Releasing calibrated-explanations on PyPI
=========================================

This is a guide on how to release calibrated-explanations on [PyPI].


## Pre-requisites (once)

First make sure to understand the contents of [official Python packaging guide].

[official Python packaging guide]: https://packaging.python.org/en/latest/tutorials/packaging-projects/

Make sure you have the build and twine Python packages installed.
Use pip or your package manager of choice to install them.
Make sure you have a [test.pypi.org] and [pypi.org] accounts.
Set-up your `~/.pypirc` as instructed when creating a token:

```ini
[testpypi]
  username = __token__
  password = pypi-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

## Release steps (every time)

1. Checkout the main branch.

	```bash
	$ git checkout main
	$ git pull
	```

2. Make sure the build is green on the `main` branch

3. Bump the version number on `pyproject.toml`.

4. In the project root, clear up your `dist/` directory.

	```bash
	$ rm -r dist/
	```

5. Build a `dist/` of the latest version.

	```bash
	python -m build
	```

6. Upload your package using twine.

	```bash
	twine upload --repository testpypi dist/*
	```

7. Check that the package page was rendered correctly
   using the link reported by twine, in the form:

	https://test.pypi.org/project/calibrated-explanations/

8. Test installation of the package in a virtual environment:

	```bash
	$ python -m venv venv
	$ source venv/bin/activate
	(venv) $ python3 -m pip install -r requirements.txt
	(venv) $ python3 -m pip install ipython  # TODO: move to requirements!
	(venv) $ python3 -m pip install -i https://test.pypi.org/simple/ --no-deps calibrated-explanations
	(venv) $ python3
	>>> import calibrated_explanations
	>>> ...
	```

9. Commit and tag a new [semantic version] on git.

	```bash
	git add .
	git commit -m 'calibrated-explanations vX.Y.Z'
	git tag vX.Y.Z
	git push
	git push --tags
	```

10. Upload your package to the real PyPI using twine:

	```bash
	twine upload pypi dist/*
	```

11. check that the project page was rendered correctly in:

	https://pypi.org/project/calibrated-explanations/

12. test installation

	```bash
	$ pip install calibrated-explanations
	>>> import calibrated_explanations
	```

This could be automated through CI,
but would only be worth it if we are releasing very often.
So for now, we just follow the guide.

[semantic version]: https://semver.org/
[test.pypi.org]: https://test.pypi.org/
[pypi.org]: https://pypi.org/
[PyPI]: https://pypi.org/

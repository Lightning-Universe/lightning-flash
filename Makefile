.PHONY: test clean docs

# assume you have installed need packages
export SPHINX_MOCK_REQUIREMENTS=1

test: clean
	pip install -q -r requirements.txt
	pip install -q -r requirements/test.txt
	# install APEX, see https://github.com/NVIDIA/apex#linux

	# use this to run tests
	python -m coverage run --source flash -m pytest flash tests -v --flake8
	python -m coverage report

docs: clean
	pip install --quiet -r requirements/docs.txt
	python -m sphinx -b html -W --keep-going docs/source docs/build

clean:
	rm -rf _ckpt_*
	rm -rf ./lightning_logs
	# clean all temp runs
	rm -rf $(shell find . -name "mlruns")
	rm -rf .mypy_cache
	rm -rf .pytest_cache
	rm -rf **/__pycache__
	rm -rf ./docs/build
	rm -rf ./docs/source/**/generated

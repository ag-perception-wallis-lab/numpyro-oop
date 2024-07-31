.PHONY: update-deps update-and-install

update-deps:
	pipenv install
	pipenv requirements > requirements.txt
	pipenv requirements --dev-only > requirements-dev.txt
	sed '1d' requirements.txt > requirements.tmp && mv requirements.tmp requirements.txt
	sed '1d' requirements-dev.txt > requirements-dev.tmp && mv requirements-dev.tmp requirements-dev.txt

update-and-install: update-deps
	pipenv run pip install -e ".[dev]"

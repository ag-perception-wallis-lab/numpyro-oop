update-deps:
	pipenv install
	pipenv requirements > requirements.txt
	pipenv requirements --dev-only > requirements-dev.txt
	sed -i '1d' requirements.txt
	sed -i '1d' requirements-dev.txt

update-and-install:
	update-deps
	pipenv run pip install -e ".[dev]"

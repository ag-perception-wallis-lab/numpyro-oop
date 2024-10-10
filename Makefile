REQUIREMENTS = requirements.txt requirements-dev.txt Pipfile.lock

.PHONY: clean-up-deps extract-deps sanitize-deps upgrade install

all: upgrade install

clean-up-deps:
	@for file in $(REQUIREMENTS); do \
		if [ -e $$file ]; then \
			echo "Removing '$$file'..."; \
			rm $$file; \
		else \
			echo "File '$$file' does not exist."; \
		fi \
	done

extract-deps:
	pipenv lock
	pipenv requirements > requirements.txt
	pipenv requirements --dev-only > requirements-dev.txt

sanitize-deps:
	sed '1d' requirements.txt > requirements.tmp && mv requirements.tmp requirements.txt
	sed '1d' requirements-dev.txt > requirements-dev.tmp && mv requirements-dev.tmp requirements-dev.txt

upgrade: clean-up-deps extract-deps sanitize-deps

install:
	pipenv run pip install -e ".[dev]"

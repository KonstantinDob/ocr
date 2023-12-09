NAME ?= ocr
GPUS ?= all

.PHONY: build
build:
	docker build -t $(NAME) .

.PHONY: run
run:
	docker run --rm -it \
		-v $(shell pwd):/workdir \
		--name=$(NAME) \
		$(NAME)

.PHONY: stop
stop:
	-docker stop $(NAME)
	-docker rm $(NAME)

.PHONY: style
style:
	git config --global --add safe.directory ./ && pre-commit run --verbose --files ocr/*

.PHONY: test-cov
test-cov:
	docker run --rm \
		-v $(shell pwd):/workdir \
		--name=$(NAME) \
		$(NAME) \
		pytest \
			-p no:logging \
			--cache-clear \
			--cov ocr/builder \
			--cov ocr/model \
			--junitxml=pytest.xml \
			--cov-report term-missing:skip-covered \
			--cov-report xml:coverage.xml \
			tests

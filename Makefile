NAME ?= ocr
GPUS ?= all

build:
	docker build -t $(NAME) .

run:
	docker run --rm -it \
		--gpus=$(GPUS) \
		-v $(shell pwd):/workdir \
		--name=$(NAME) \
		$(NAME)

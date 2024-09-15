##################################################################
# ENV VARS
##################################################################
export GID := $(shell id -g)
export UID := $(shell id -u)
export BRANCH := $(shell git rev-parse --abbrev-ref HEAD)
export COMMIT_ID := $(shell git rev-parse --short=9 HEAD)
export BUILD_TIME := $(shell date '+%Y%m%d')
export COMMIT_BODY := $(shell git rev-list --format=%B --max-count=1 HEAD | tail +2)
export USERNAME := $(ACCOUNT_NAME)
export PASSWORD := $(ACCOUNT_PASSWORD)
ifeq "$(BRANCH)" "dev"
	DOCKER_NAMESPACE := dev
else ifeq "$(BRANCH)" "master"
	DOCKER_NAMESPACE := prod
else
	DOCKER_NAMESPACE := stage
endif
export DOCKER_NAMESPACE := $(DOCKER_NAMESPACE)

# check username and password
ifeq ($(USERNAME),)
$(warning USERNAME is not set)
endif
ifeq ($(PASSWORD),)
$(warning PASSWORD is not set)
endif

pre-build:
	@echo "DOCKER_NAMESPACE: $(DOCKER_NAMESPACE)"
	@echo "GID: $(GID)"
	@echo "UID: $(UID)"
	@echo "BRANCH: $(BRANCH)"
	@echo "COMMIT_ID: $(COMMIT_ID)"
	@echo "BUILD_TIME: $(BUILD_TIME)"
	@echo "USERNAME: $(USERNAME)"
	@echo "PASSWORD: $(PASSWORD)"

##################################################################
# COMMANDS
##################################################################
build-cuda-ros-dev: pre-build
	docker compose -f docker-compose.yml build cuda-ros-dev

start-cuda-ros-dev:pre-build
	docker compose -f docker-compose.yml up -d cuda-ros-dev

push-cuda-ros-dev:pre-build
	COMMIT_ID=latest docker compose -f docker-compose.yml push cuda-ros-dev
	docker compose -f docker-compose.yml push cuda-ros-dev

stop-cuda-ros-dev:pre-build
	docker compose -f docker-compose.yml stop cuda-ros-dev
	docker compose -f docker-compose.yml rm cuda-ros-dev

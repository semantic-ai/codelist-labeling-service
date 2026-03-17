FROM semtech/mu-python-template:feature-fastapi
LABEL maintainer="joachim@ml2grow.com"

ENV BASE_REGISTRY_URI=https://api.basisregisters.vlaanderen.be

ADD ./decide_ai_service_base-0.1.0-py3-none-any.whl .
RUN uv pip install ./decide_ai_service_base-0.1.0-py3-none-any.whl
FROM semtech/mu-python-template:feature-fastapi
LABEL maintainer="joachim@ml2grow.com"

ADD ./decide_ai_service_base-0.1.12-py3-none-any.whl .
RUN uv pip install decide_ai_service_base-0.1.12-py3-none-any.whl
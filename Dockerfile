FROM semtech/mu-python-template:feature-fastapi
LABEL maintainer="joachim@ml2grow.com"

# SPARQL statements are large, multiline records. Keep them opt-in so normal
# container logs remain readable; deployments can override either setting.
ENV LOG_LEVEL=INFO \
    LOG_SPARQL_ALL=false

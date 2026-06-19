import logging
import os
from threading import Lock

from helpers import query, log
from escape_helpers import sparql_escape_uri

from fastapi import APIRouter, BackgroundTasks

from decide_ai_service_base.task import Task
from decide_ai_service_base.util import wait_for_triplestore, TaskProcessor, process_open_tasks, fail_busy_and_scheduled_tasks, write_agent_info
from decide_ai_service_base.schema import NotificationResponse, TaskOperationsResponse
from src.task import ModelAnnotatingTask, ModelBatchAnnotatingTask, ClassifierTrainingTask, ImpactAssessmentTask, ClassifierAnnotatingTask

# Configure root logger level from LOG_LEVEL env var.
# The mu-python-template only configures its own 'MU_PYTHON_TEMPLATE_LOGGER',
# but task classes use per-class loggers (e.g. 'ModelAnnotatingTask') that
# inherit from the root logger. Without this, INFO/DEBUG messages are dropped.
_log_level = os.environ.get("LOG_LEVEL", "WARNING").upper()
logging.basicConfig(level=getattr(logging, _log_level, logging.WARNING))

_open_tasks_lock = Lock()

@app.on_event("startup")
async def startup_event():
    wait_for_triplestore()
    fail_busy_and_scheduled_tasks()
    write_agent_info("http://lblod.data.gift/id/components/codelist-labeling/v1.0.0", "impact_annotator")
    write_agent_info("http://lblod.data.gift/id/components/codelist-labeling/v1.0.0", "model_annotator")
    write_agent_info("http://lblod.data.gift/id/components/codelist-labeling/v1.0.0", "classifier_annotator")

    process_open_tasks(_open_tasks_lock)


router = APIRouter()


@router.post("/delta", status_code=202)
async def delta(background_tasks: BackgroundTasks) -> NotificationResponse:
    processor = TaskProcessor(_open_tasks_lock)
    background_tasks.add_task(processor)
    return NotificationResponse(status="accepted", message="Processing started")


@router.get("/task/operations")
def get_task_operations() -> TaskOperationsResponse:
    return TaskOperationsResponse(
        task_operations=[
            clz.__task_type__ for clz in Task.supported_operations()
        ]
    )


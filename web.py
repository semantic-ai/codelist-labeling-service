from helpers import query, log
from escape_helpers import sparql_escape_uri

from fastapi import APIRouter, BackgroundTasks

from decide_ai_service_base.task import Task
from decide_ai_service_base.util import wait_for_triplestore, process_open_tasks, fail_busy_and_scheduled_tasks
from decide_ai_service_base.schema import NotificationResponse, TaskOperationsResponse
from src.task import ModelAnnotatingTask, ModelBatchAnnotatingTask, ClassifierTrainingTask, ImpactAssessmentTask

@app.on_event("startup")
async def startup_event():
    wait_for_triplestore()
    fail_busy_and_scheduled_tasks()
    process_open_tasks()


router = APIRouter()


@router.post("/delta", status_code=202)
async def delta(background_tasks: BackgroundTasks) -> NotificationResponse:
    background_tasks.add_task(process_open_tasks)
    return NotificationResponse(status="accepted", message="Processing started")


@router.get("/task/operations")
def get_task_operations() -> TaskOperationsResponse:
    return TaskOperationsResponse(
        task_operations=[
            clz.__task_type__ for clz in Task.supported_operations()
        ]
    )


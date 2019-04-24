import os

from celery import Celery

from config import raise_error

broker = os.getenv('CELERY_BROKER_URL') or raise_error('CELERY_BROKER_URL')
backend = os.getenv('CELERY_RESULT_BACKEND') or raise_error('CELERY_RESULT_BACKEND')
app = Celery('test_celery',
             broker=broker,
             backend=backend,
             include=['tasks'])

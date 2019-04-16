from celery import Celery

# app = Celery('IPS')
# app = app.config_from_object('indoor_positioning.config.CeleryConfig')
app = Celery('test_celery',
             broker='amqp://guest:guest@localhost/ips',
             backend='rpc://',
             include=['tasks'])

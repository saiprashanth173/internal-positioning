## Instructions project setup

Install the puthon packages needed in requirement.txt

Install [postgres](https://www.postgresql.org/download/) 
and [rabbitmq](https://www.rabbitmq.com/download.html)
```bash
$ rabbitmqctl add_vhost ips
$ rabbitmqctl add_user <user> <password>
$ rabbitmqctl set_permissions -p ips <user> ".*" ".*" ".*"

$ export CELERY_BROKER_URL=amqp://<user>:<password>@localhost:5672/ips
$ export CELERY_RESULT_BACKEND=amqp://<user>:<password>@localhost:5672/ips


$ export PG_USER=<postgres-user>
$ export PG_PASSWORD=<postgres-password>
$ export PG_HOST=<hostname> # localhost 
$ export PG_DB=ips

# For creating tables
createdb ips
psql -U <user> -W -d ips < migrations.sql

```

#### Run producer and celery processor
```bash
# These should be running continuously
$ celery  -A celery_worker  worker --loglevel=info
$ python producer.py # In another terminal, you might need to export env variables again
``` 

#### Run server
```bash
python server.py <host> <post> # In another terminal, you might need to export env variables again
#For now use <host>=localhost port=<8080>

```

import json

from bottle import static_file, route, run, get
from bottle_websocket import GeventWebSocketServer
from bottle_websocket import websocket
from gevent import sleep
from geventwebsocket import WebSocketError

import generators
from config import GENERATOR


@route('/static/<filename>')
def server_static_html(filename):
    return static_file(filename, root='./html/')


@route('/static/css/<filename>')
def server_static(filename):
    return static_file(filename, root='./css/')


@route('/static/js/<filename>')
def server_static(filename):
    return static_file(filename, root='./js/')


@get('/websocket', apply=[websocket])
def echo(ws):
    counter = 0
    Generator = getattr(generators, GENERATOR)
    generator = Generator()
    while True:
        try:
            data = generator.get_next()
            ws.send(json.dumps(data.to_json(orient='records')))
            sleep(1)
            counter += 1
        except WebSocketError:
            break


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='server for handling websockets')
    parser.add_argument('host', type=str, help="host name")
    parser.add_argument('port', type=str, help="host name")
    args = parser.parse_args()
    HOST = args.host
    PORT = args.port

    run(host=HOST, port=PORT, server=GeventWebSocketServer)

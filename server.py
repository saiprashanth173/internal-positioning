import json

from bottle import static_file, route, run, get
from bottle_websocket import GeventWebSocketServer
from bottle_websocket import websocket
from gevent import sleep
from geventwebsocket import WebSocketError

import generators
from config import GENERATOR, MULTI_GENERATOR


@route('/static/<filename>')
def server_static_html(filename):
    return static_file(filename, root='./static/html/')


@route('/static/css/<filename>')
def server_static(filename):
    return static_file(filename, root='./static/css/')


@route('/static/js/<filename>')
def server_static(filename):
    return static_file(filename, root='./static/js/')


@route('/static/lib/js/<filename>')
def server_static(filename):
    return static_file(filename, root='./static/lib/js')


@route('/static/lib/css/<filename>')
def server_static(filename):
    return static_file(filename, root='./static/lib/css')


@route('/static/assets/<filename>')
def server_static(filename):
    return static_file(filename, root='./static/assets/')


@get('/easy', apply=[websocket])
def echo(ws):
    Generator = getattr(generators, GENERATOR)
    generator = Generator()
    while True:
        try:
            data = generator.get_next()
            ws.send(json.dumps(data.to_json(orient='records')))
            sleep(1)
        except WebSocketError:
            break


@get('/multibuilding', apply=[websocket])
def multiple_building(ws):
    Generator = getattr(generators, MULTI_GENERATOR)
    generator = Generator()
    while True:
        try:
            data = generator.get_next()
            ws.send(json.dumps(data.to_json(orient='records')))
            sleep(0.005)
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

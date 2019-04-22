import json

from bottle import static_file, route, run, get, request
from bottle_websocket import GeventWebSocketServer
from bottle_websocket import websocket
from gevent import sleep
from geventwebsocket import WebSocketError

import generators
from config import GENERATOR, MULTI_GENERATOR, PG_GENERATOR


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
def easy(ws):
    Generator = getattr(generators, GENERATOR)
    generator = Generator()
    while True:
        try:
            data = generator.get_next()
            ws.send(json.dumps(data.to_json(orient='records')))
            sleep(1)
        except WebSocketError:
            break


@get('/medium', apply=[websocket])
def medium(ws):
    Generator = getattr(generators, MULTI_GENERATOR)
    generator = Generator()
    while True:
        try:
            data = generator.get_next()
            ws.send(json.dumps(data.to_json(orient='records')))
            sleep(1)
        except WebSocketError:
            break


@get('/hard', apply=[websocket])
def hard(ws):
    Generator = getattr(generators, PG_GENERATOR)
    generator = Generator()
    while True:
        try:
            user_id = request.query.get('user_id', None)
            data = generator.get_next(user_id)
            ws.send(json.dumps(data))
            sleep(1)
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

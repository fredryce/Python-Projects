#!/usr/bin/env python3.7

import logging
import asyncio
import socket

async def handle_client(reader, writer):
	request = None
	while request != 'quit':
		request = (await reader.read(255)).decode('utf8')
		print("got request for ", request)

		try:
			response = str(eval(request)) + '\n'
		except Exception:
			response = request
		writer.write(response.encode('utf8'))
		await writer.drain()
	writer.close()

loop = asyncio.get_event_loop()
loop.create_task(asyncio.start_server(handle_client, 'localhost', 15555))
loop.run_forever()

'''

async def handle_client(client):
	request = None
	while request != 'quit':
		request = (await loop.sock_recv(client, 255)).decode('utf8')
		response = str(eval(request)) + '\n'
		await loop.sock_sendall(client, response.encode('utf8'))
	client.close()

async def run_server():
	while True:
		client, _ = await loop.sock_accept(server)
		loop.create_task(handle_client(client))

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(('localhost', 15555))
server.listen(8)
server.setblocking(False)

loop = asyncio.get_event_loop()
loop.run_until_complete(run_server())

'''
from socket import *
import asyncio


loop = asyncio.get_event_loop() #define event loop to handle concurrency


async def echo_sever(address):
	sock =socket(AF_INET, SOCK_STREAM) #define socket
	sock.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)  #setting socket option
	sock.bind(address)  #bind the socket to ip and port
	sock.listen(5)
	sock.setblocking(False)
	while True:
		client, addr = await loop.sock_accept(sock) # await handle the coroutines
		print('Connected from', addr)
		loop.create_task(echo_handler(client))


async def echo_handler(client):
	with client:
		while True:
			data = await loop.sock_recv(client, 10000)
			if not data:
				break
			await loop.sock_sendall(client, b'got: ' + data)


loop.create_task(echo_sever(('', 25000)))
loop.run_forever()
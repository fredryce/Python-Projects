import asyncio

async def say(what, when):
	await asyncio.sleep(when)
	print(what)

loop = asyncio.get_event_loop()
t1 = loop.create_task(say("first", 0.1))
t2 = loop.create_task(say("second", 0.1))
loop.run_until_complete(asyncio.gather(t1, t2))

print(type(asyncio.gather(t1,t2)))
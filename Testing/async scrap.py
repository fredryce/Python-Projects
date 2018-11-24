import asyncio
import aiohttp
import json
import time
from bs4 import BeautifulSoup as soup

def timer(function):
	def x():
		start = time.time()
		function()
		total = time.time() - start
		print("time total took", total)
	return x



async def get_status(websites):
	#async need context manager?
	async with aiohttp.ClientSession() as session:






		'''
		statuses = {}
		tasks = [get_website_status(website, session) for website in websites]
		for status in await asyncio.gather(*tasks):
			if not statuses.get(status):
				statuses[status] = 0
			statuses[status] += 1
		print(json.dumps(statuses))
		'''


async def get_website_status(url, session):
	#async need context manager? 
	async with session.get(url) as response:
		return response.status


async def get_all_sites(baseUrl, session):
	async with session.get(baseUrl) as response:
		container = soup(response, "html.parser")
		print(container)



@timer
def myfunction():
	websites = ["https://www.google.com", "https://www.youtube.com"]
	#pool = concurrent.futures.ProcessPoolExecutor()
	loop = asyncio.get_event_loop()		
	#loop.run_in_executor(pool, get_status(websites), 2000)
	loop.run_until_complete(get_status(websites))



if __name__ =="__main__":
	myfunction()
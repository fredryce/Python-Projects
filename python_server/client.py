import requests


data = "wtf is going on"

with requests.Session() as sess:
	r = sess.post("https://xin-6inw.localhost.run", data)
	print(r.content)
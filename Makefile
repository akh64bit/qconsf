default: ci

ci: build
	echo "Success"

build: 
	docker build --no-cache -t sf:2 .

run:
	docker run -p 8888:8888 -it sf:1

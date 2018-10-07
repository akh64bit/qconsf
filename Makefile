default: ci

ci: build
	echo "Success"

build: 
	docker build -t sf:1 .

run:
	docker run -p 8888:8888 -it sf:1

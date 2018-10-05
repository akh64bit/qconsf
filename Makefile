default: ci

ci: build
	echo "Success"

build: 
	docker build -t sf:1 .
.PHONY: up

up:
	jupyter notebook --config jupyter/config.py

install:
	pip install -r requirements.txt

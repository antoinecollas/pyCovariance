#!/bin/sh
mkdir data
aria2c --dir=./data --input-file=./urls.txt -x 16

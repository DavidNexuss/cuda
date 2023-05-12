#!/bin/sh
cd backends
make
cd ..
./run.sh $@ scripts/runFrameGL.c backends/backend.a backends/gl.c -lGL -lGLEW -lglfw -lassimp;

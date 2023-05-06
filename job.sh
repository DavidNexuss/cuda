#!/bin/sh

export PATH=/Soft/cuda/11.2.1/bin:$PATH
nsys profile --trace=cuda ./tracer

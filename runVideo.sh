#!/bin/sh
./run.sh ext/sceneEncode.c scripts/runVideo.c -lavcodec -lavutil -lavfilter -lswscale $@

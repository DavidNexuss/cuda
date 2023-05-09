#!/bin/sh
./runBoada.sh ext/sceneEncode.c scripts/runVideo.c -lavcodec -lavutil -lavfilter -lswscale $@

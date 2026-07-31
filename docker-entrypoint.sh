#!/bin/sh
set -e

if [ "$(id -u)" = "0" ]; then
    if [ -d /data ]; then
        chown -R tinycontext:tinycontext /data 2>/dev/null || true
    fi

    if [ -d /config ]; then
        chown -R tinycontext:tinycontext /config 2>/dev/null || true
    fi

    exec gosu tinycontext "$@"
fi

exec "$@"

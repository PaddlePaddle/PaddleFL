#!/bin/bash

echo "Stop service!"

ps -ef | grep -E "fl" | grep -v grep | awk '{print $2}' | xargs kill -9

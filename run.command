#!/bin/bash
cd "$(dirname "$BASH_SOURCE")"/makestuff || {
    echo "Error getting script directory" >&2
    exit 1
}
cmake ..
make -j12
cd ..
./makestuff/clpath

#!/usr/bin/env bash

# set up environment (optional)
FILE=~/.zshrc
if test -f "$FILE"; then
    source $FILE
fi

# if any command inside script returns error, exit and return that error
set -e

# magic line to ensure that we're always inside the root of our application,
# no matter from which directory we'll run script
# thanks to it we can just enter `./scripts/run-tests.bash`
cd "${0%/*}/.."

# let's fake failing test for now
echo "Running tests"
echo "............................"
#echo "Failed!" && exit 1

# example of commands for different languages
python ./tests/run_tests.py -bv

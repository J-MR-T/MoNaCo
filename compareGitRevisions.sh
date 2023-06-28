#!/usr/bin/env bash

# This script checks out two git revisions ($1, $2), build the monaco target $3, and compares the output of the monaco command arguments ($4, $5, ...) between the two revisions.

# Usage: ./compareGitRevisions.sh <revision1> <revision2> <make target> <monaco command>

rev1="$(git rev-parse $1)"
rev1_orig="$1"
rev2="$(git rev-parse $2)"
rev2_orig="$2"
makeTarget="$3"

shift 3

# if rev1 or 2 are empty, or if rev1 is -h or --help, print usage
if [ -z "$rev1" ] || [ -z "$rev2" ] || [ -z "$makeTarget" ] || [ "$rev1" == "-h" ] || [ "$rev1" == "--help" ]; then
    echo "Usage: $0 <revision1> <revision2> <make target> <monaco command>"
    exit 1
fi

# if rev1 or rev2 are not valid git revisions, print error
if ! git rev-parse "$rev1" >/dev/null 2>&1; then
    echo "Error: $rev1 is not a valid git revision"
    exit 1
fi

if ! git rev-parse "$rev2" >/dev/null 2>&1; then
    echo "Error: $rev2 is not a valid git revision"
    exit 1
fi

# if make target is not valid, print error
make -q "$makeTarget" >/dev/null 2>&1
if [ $(echo $?) -gt 1 ]; then
    echo "Error: $makeTarget is not a valid make target"
    exit 1
fi

checkoutAndRun () {
    rev="$1"
    rev_orig="$2"
    git checkout "$rev" >/dev/null 2>&1
    shift 2

    tmpBuildDir="moreBuilds/build_$rev"
    echo "Running \`MONACO_BUILD_DIR=$tmpBuildDir make -e $makeTarget\`"
    MONACO_BUILD_DIR="$tmpBuildDir" make -e "$makeTarget" >/dev/null 2>&1
    if [ $(echo $?) -gt 0 ]; then
        echo "Error: $makeTarget failed to build"
    fi

    # run monaco command
    echo "Running \`monaco \"$@\" on $rev_orig\`"
    "$tmpBuildDir/bin/monaco" "$@"
}

# save current revision
currentRev=$(git rev-parse HEAD)

checkoutAndRun "$rev1" "$rev1_orig" "$@"

checkoutAndRun "$rev2" "$rev2_orig" "$@"

# checkout current revision
git checkout "$currentRev" >/dev/null 2>&1

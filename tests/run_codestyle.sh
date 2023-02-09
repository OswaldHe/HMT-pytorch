#!/usr/bin/env bash

pip install flake8==6.0.0

for ARGUMENT in "$@"; do

    KEY=$(echo $ARGUMENT | cut -f1 -d=)
    VALUE=$(echo $ARGUMENT | cut -f2 -d=)

    case "$KEY" in
    DIFF_BRANCH) DIFF_BRANCH=${VALUE} ;;
    MODE) MODE=${VALUE} ;;
    *) ;;
    esac
done

if [[ "$DIFF_BRANCH" == "" ]]; then
    DIFF_BRANCH="dev"
fi

PTH="lm_experiments_tools"
echo "running flake8 on package $PTH"

res=$(git diff --cached --name-only --diff-filter=ACMR origin/$DIFF_BRANCH $PTH | grep \.py\$ | tr -d "[:blank:]")
if [ -z "$res" ]
then
  exit 0
else
  flake8 --statistics --count $(git diff --cached --name-only --diff-filter=ACMR origin/$DIFF_BRANCH $PTH | grep \.py\$)
fi
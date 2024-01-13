#!/bin/bash
set -e
cargo install --force honggfuzz --version "^0.5.47"
for TARGET in fuzz_targets/*; do
    FILENAME=$(basename $TARGET)
	FILE="${FILENAME%.*}"
	if [ -d hfuzz_input/$FILE ]; then
	    HFUZZ_INPUT_ARGS="-f hfuzz_input/$FILE/input"
	fi
	HFUZZ_BUILD_ARGS="--features honggfuzz_fuzz" HFUZZ_RUN_ARGS="--run_time 30 --exit_upon_crash -v $HFUZZ_INPUT_ARGS" cargo hfuzz run $FILE

	if [ -f hfuzz_workspace/$FILE/HONGGFUZZ.REPORT.TXT ]; then
		cat hfuzz_workspace/$FILE/HONGGFUZZ.REPORT.TXT
		for CASE in hfuzz_workspace/$FILE/SIG*; do
			cat $CASE | xxd -p
		done
		exit 1
	fi
done

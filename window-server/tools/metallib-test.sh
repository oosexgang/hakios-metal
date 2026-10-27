#!/bin/sh
# The Metal plug-in's reading of a .metallib (userland/src/hakios_metallib.c) on this machine, under the address and
# undefined-behaviour sanitizers: a file made here and cut and bent every way (each refused, nothing read outside it), then
# every .metallib under DIR read by it and by tools/metallib.py, which must agree on every function.
# usage: tools/metallib-test.sh [DIR]     DIR: an extracted system (sysimage/rootfs); without one, only the made file
set -e
cd "$(dirname "$0")/.."
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
cc -O1 -g -Wall -Wextra -fsanitize=address,undefined -fno-sanitize-recover=all -Iuserland/src \
	tools/metallib_test.c userland/src/hakios_metallib.c -o "$tmp/metallib_test"
"$tmp/metallib_test" cuts
[ -n "$1" ] || exit 0
n=0
other=0
find "$1" -name '*.metallib' -type f > "$tmp/files"
while IFS= read -r f; do
	if ! python3 tools/metallib.py list "$f" > "$tmp/py.all" 2>/dev/null; then
		# not an MTLB container (the GPU driver's own runtime libraries are another format): the C reading must refuse it too
		if "$tmp/metallib_test" list "$f" > "$tmp/c"; then
			echo "metallib-test: $f: tools/metallib.py cannot read it and the C reading took it"
			exit 1
		fi
		other=$((other + 1))
		continue
	fi
	sed -n '/ functions$/,$p' "$tmp/py.all" > "$tmp/py"
	"$tmp/metallib_test" list "$f" > "$tmp/c" || true
	if ! cmp -s "$tmp/py" "$tmp/c"; then
		echo "metallib-test: $f: the two readings differ"
		diff "$tmp/py" "$tmp/c" | head -5
		exit 1
	fi
	n=$((n + 1))
done < "$tmp/files"
echo "metallib-test: $n files under $1: the C reading and tools/metallib.py agree on every function; $other that are not MTLB refused by both"

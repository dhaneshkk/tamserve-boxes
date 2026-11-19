#!/bin/bash
# End-to-end test with a real RTSP server that needs a password: reconnects, health metrics, passwords kept off
# command lines, and a coordinator moving cameras off a box that is killed. Needs: mediamtx, ffmpeg, curl, python3,
# OpenVINO's C library on LD_LIBRARY_PATH, and the release binary. Run from rust/:
#   scripts/rtsp_test.sh MODEL.onnx VIDEO.mp4
# MODEL: a 1-channel YOLOX (tamserve-boxes fold); VIDEO: any short H.264 video. Exits non-zero on any failure.
set -euo pipefail
MODEL=$(realpath "$1"); VIDEO=$(realpath "$2")
B=$(realpath target/release/tamserve-boxes)
MTX=${MEDIAMTX:-mediamtx}
W=$(mktemp -d); cd "$W"; trap 'pkill -P $$ 2>/dev/null || true; kill $(jobs -p) 2>/dev/null || true' EXIT
PW="pw-$(head -c 9 /dev/urandom | base64 | tr -dc a-zA-Z0-9)"
cat > mediamtx.yml <<YML
logLevel: warn
rtspAddress: :18554
rtmp: no
hls: no
webrtc: no
srt: no
api: no
metrics: no
authInternalUsers:
  - user: admin
    pass: $PW
    permissions: [{action: publish}, {action: read}]
paths:
  all_others:
YML
printf '%s\n' "$PW" > good.pw; printf 'wrong\n' > bad.pw; chmod 600 good.pw bad.pw
head -c 48 /dev/urandom | base64 > token; chmod 600 token
publish() { for i in 1 2 3 4; do ffmpeg -nostdin -loglevel error -re -stream_loop -1 -i "$VIDEO" -c copy -f rtsp \
            -rtsp_transport tcp "rtsp://admin:$PW@127.0.0.1:18554/c$i" & done; }
box_config() {  # $1: HTTP port, $2: extra camera lines
  printf '[detector]\nmodel = "%s"\nthreads = 1\n[decode]\ndecoder = "software"\nstall_seconds = 4\nbackoff_first = 0.5\nbackoff_max = 4\n[http]\nlisten = "127.0.0.1:%s"\n' "$MODEL" "$1"
  for i in 1 2 3 4; do printf '[[camera]]\nname = "c%s"\nurl = "rtsp://127.0.0.1:18554/c%s"\nuser = "admin"\npassword_file = "good.pw"\n' $i $i; done
  printf '%b' "${2:-}"
}
status() { curl -sf "localhost:$1/status"; }
check() {  # $1: a Python condition on s, $2: the status JSON, $3: what is being checked
  python3 -c "import json,sys; s=json.loads(sys.argv[1]); assert $1, sys.argv[1][:600]" "$2" || { echo "FAIL: $3"; exit 1; }
  echo "ok: $3"
}

"$MTX" mediamtx.yml & sleep 1; publish; sleep 2
box_config 19301 '[[camera]]\nname = "wrong"\nurl = "rtsp://127.0.0.1:18554/c1"\nuser = "admin"\npassword_file = "bad.pw"\n' > solo.toml
"$B" run --config solo.toml > solo.jsonl 2> solo.log & SOLO=$!
sleep 8
check "all(s['cameras'][c]['state'] == 'up' for c in ['c1','c2','c3','c4'])" "$(status 19301)" "four cameras up over RTSP with a password"
check "s['cameras']['wrong']['state'] == 'down' and 'nauthor' in s['cameras']['wrong']['last_error']" "$(status 19301)" "a wrong password is reported and retried"
! ps -eo args | grep -v -e ffmpeg -e grep -e mediamtx | grep -q "$PW" || { echo "FAIL: the password is on a command line"; exit 1; }
echo "ok: no password on any tamserve-boxes command line"
curl -sf localhost:19301/metrics | grep -q 'tamserve_camera_up{camera="c1"} 1' && echo "ok: Prometheus metrics"
curl -sf -X POST localhost:19301/v1/chat/completions -d '{"model":"c1","messages":[{"role":"user","content":"?"}]}' \
  | python3 -c "import json,sys; r=json.load(sys.stdin); assert r['object']=='chat.completion'; json.loads(r['choices'][0]['message']['content'])" \
  && echo "ok: OpenAI-compatible /v1/chat/completions"
pkill -f "mediamtx mediamtx.yml"; sleep 6
check "all(s['cameras'][c]['state'] in ('down','connecting') for c in ['c1','c2'])" "$(status 19301)" "cameras go down with the server"
"$MTX" mediamtx.yml & sleep 1; pkill -f "ffmpeg.*18554" || true; sleep 0.5; publish; sleep 10
check "all(s['cameras'][c]['state'] == 'up' and s['cameras'][c]['reconnects'] > 0 for c in ['c1','c2','c3','c4'])" "$(status 19301)" "and come back by themselves"
kill $SOLO

printf 'listen = "127.0.0.1:17400"\ntoken_file = "token"\ndead_seconds = 3\nhttp = "127.0.0.1:19400"\n' > coord.toml
for i in 1 2 3 4; do printf '[[camera]]\nname = "c%s"\n' $i >> coord.toml; done
box_config 19302 > a.toml; box_config 19303 > b.toml
"$B" coordinator --config coord.toml > results.jsonl 2> coord.log &
"$B" box --config a.toml --coordinator 127.0.0.1:17400 --token-file token --name box-a --capacity 4 > /dev/null 2> a.log & A=$!
"$B" box --config b.toml --coordinator 127.0.0.1:17400 --token-file token --name box-b --capacity 4 > /dev/null 2> b.log &
sleep 12
check "sorted(len(b['cameras']) for b in s['boxes'].values()) == [2, 2]" "$(status 19400)" "four cameras spread over two boxes"
KILLED_MS=$(($(date +%s%N) / 1000000)); kill -9 $A; sleep 8
check "list(s['boxes']) == ['box-b'] and len(s['boxes']['box-b']['cameras']) == 4" "$(status 19400)" "a killed box's cameras move to the other"
python3 - "$KILLED_MS" <<'PY'
import json, sys, collections
kill = int(sys.argv[1]); by = collections.defaultdict(list)
for line in open("results.jsonl"):
    r = json.loads(line); by[r["camera"]].append(r["time_ms"])
worst = max(min((t for t in v if t > kill), default=1e18) - max((t for t in v if t <= kill), default=kill) for v in by.values()) / 1000
assert worst < 6, f"a camera went {worst:.1f} s without results"
print(f"ok: the longest gap in any camera's results across the kill: {worst:.1f} s")
PY
echo "ALL PASSED"

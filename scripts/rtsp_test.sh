#!/bin/bash
# End-to-end test with a real RTSP server that needs a password: reconnects, health metrics, passwords kept off
# command lines, a coordinator moving cameras off a box that is killed, and one connection per camera shared by the
# detector, the recorder and live viewers. Needs: mediamtx, ffmpeg, curl, python3,
# OpenVINO's C library on LD_LIBRARY_PATH, and the release binary. Run from the repository's root:
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
# check, but waiting up to $4 seconds (default 30) for it to hold: what a fixed sleep guesses at
# varies with the machine and, on the first run after a build, with OpenVINO compiling the model.
eventually() {  # $1: a Python condition on s, $2: the box's HTTP port, $3: what is being checked
  local end=$((SECONDS + ${4:-30}))
  until python3 -c "import json,sys; s=json.loads(sys.argv[1]); assert $1" "$(status "$2" || echo '{}')" 2>/dev/null; do
    [ "$SECONDS" -lt "$end" ] || { check "$1" "$(status "$2" || echo '{}')" "$3"; return; }
    sleep 0.5
  done
  echo "ok: $3"
}

"$MTX" mediamtx.yml & sleep 1; publish; sleep 2
box_config 19301 '[[camera]]\nname = "wrong"\nurl = "rtsp://127.0.0.1:18554/c1"\nuser = "admin"\npassword_file = "bad.pw"\n' > solo.toml
"$B" run --config solo.toml > solo.jsonl 2> solo.log & SOLO=$!
eventually "all(s['cameras'][c]['state'] == 'up' for c in ['c1','c2','c3','c4'])" 19301 "four cameras up over RTSP with a password"
eventually "s['cameras']['wrong']['state'] == 'down' and 'nauthor' in (s['cameras']['wrong']['last_error'] or '')" 19301 "a wrong password is reported and retried"
ARGS=$(ps -eo args | grep -v -e ffmpeg -e grep -e mediamtx || true)  # not `| grep -q`: under pipefail, its early exit could pass a failure
! grep -q "$PW" <<< "$ARGS" || { echo "FAIL: the password is on a command line"; exit 1; }
echo "ok: no password on any tamserve-boxes command line"
curl -sf localhost:19301/metrics | grep -q 'tamserve_camera_up{camera="c1"} 1' && echo "ok: Prometheus metrics"
curl -sf -X POST localhost:19301/v1/chat/completions -d '{"model":"c1","messages":[{"role":"user","content":"?"}]}' \
  | python3 -c "import json,sys; r=json.load(sys.stdin); assert r['object']=='chat.completion'; json.loads(r['choices'][0]['message']['content'])" \
  && echo "ok: OpenAI-compatible /v1/chat/completions"
kill $SOLO; sleep 1
# Decode once, use twice: c1 is also encoded, from the same decode, to AV1 segments.
box_config 19301 | sed 's|^\[http\]|[encode]\nthreads = 1\npreset = 12\nbitrate_kbps = 300\nfps = 10\nwidth = 480\nheight = 320\noutput = "'"$W"'/rec/{camera}/%05d.mkv"\nsegment_seconds = 4\n\n[http]|' \
  | awk '/name = "c1"/ {print; print "encode = true"; next} {print}' > solo.toml
"$B" run --config solo.toml > solo.jsonl 2> solo.log & SOLO=$!
sleep 12
check "s['cameras']['c1']['encoded_frames'] > 50 and s['cameras']['c2']['encoded_frames'] == 0 and s['cameras']['c1']['frames_used'] > 50" "$(status 19301)" "c1 is detected and encoded from one decode"
SEG=$(ls "$W"/rec/c1/*.mkv | head -1)
[ "$(ffprobe -v error -select_streams v -show_entries stream=codec_name -of csv=p=0 "$SEG")" = av1 ] && echo "ok: c1's segments are AV1" \
  || { echo "FAIL: no AV1 segment for c1"; exit 1; }
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

# One connection, shared: c1 recorded, watched live over RTSP with its password and encoded to an AV1
# uplink copy, the motion gate on.
box_config 19304 '[motion]\n[record]\ndir = "rec"\nsegment_seconds = 2\n[encode]\nthreads = 1\nfps = 10\nwidth = 480\nheight = 320\nbitrate_kbps = 300\noutput = "up/{camera}/%05d.mkv"\nsegment_seconds = 2\n' \
  | sed 's/^name = "c1"$/name = "c1"\nrecord = true\nlive = true\nencode = true/' > rec.toml
"$B" run --config rec.toml > rec.jsonl 2> rec.log & REC=$!
sleep 6
timeout 4 curl -sf localhost:19304/live/c1.ts -o live.ts || true
sleep 4
check "s['cameras']['c1']['state'] == 'up' and s['cameras']['c1']['segments'] >= 2 and s['cameras']['c2']['segments'] == 0" "$(status 19304)" "c1 recorded over RTSP in 2 s segments, c2 not"
check "s['cameras']['c1']['encoded_frames'] > 30 and s['cameras']['c1']['frames_used'] > 30" "$(status 19304)" "c1 recorded, live and encoded at once"
fail() { echo "FAIL: $1"; exit 1; }
# curl was cut off by its timeout, so the file may end mid-packet: only its start is checked.
python3 -c "import sys; b=open('live.ts','rb').read(); assert len(b) > 8000 and b[0] == 0x47 and b[188] == 0x47, len(b)" \
  || fail "GET /live/c1.ts"
echo "ok: GET /live/c1.ts streams MPEG-TS from the box's own connection"
# It joined mid-stream: it starts at the latest keyframe, so it decodes from its first picture. (The end, cut by
# curl's timeout, may be a partial packet: only errors about missing parameter sets or references count.)
LIVE_ERRS=$(ffmpeg -v error -i live.ts -f null - 2>&1 || true)
! grep -qiE "non-existing PPS|no frame|reference" <<< "$LIVE_ERRS" || fail "the live stream started mid-picture-group: $LIVE_ERRS"
echo "ok: a viewer joining mid-stream decodes from its first picture"
[ "$(curl -s -o /dev/null -w '%{http_code}' localhost:19304/live/c2.ts)" = 404 ] || fail "c2 has a live stream"
echo "ok: a camera without live = true has no live stream"
SEG=$(ls rec/c1/*.ts | head -n -1 | tail -1)  # a closed one: the newest is still being written
FRAMES=$(ffprobe -v error -count_frames -select_streams v -show_entries stream=nb_read_frames -of csv=p=0 "$SEG" | awk 'NR == 1')
[ "$FRAMES" -gt 10 ] && [ -z "$(ffmpeg -v error -i "$SEG" -f null - 2>&1)" ] || fail "$SEG does not decode cleanly ($FRAMES frames)"
echo "ok: a closed segment decodes on its own, without errors ($FRAMES frames)"
[ "$(ls rec/c1/activity/*.jsonl | wc -l)" -ge 1 ] || fail "no activity log"
echo "ok: the activity log is written"
kill -TERM $REC; wait $REC || true
LAST=$(ls rec/c1/*.ts | tail -1)
[ -z "$(ffmpeg -v error -i "$LAST" -f null - 2>&1)" ] || fail "SIGTERM left $LAST broken"
grep -q "stopping: closing the cameras" rec.log || fail "SIGTERM was not handled"
for f in up/c1/*.mkv; do
  [ "$(ffprobe -v error -select_streams v -show_entries stream=codec_name -of csv=p=0 "$f")" = av1 ] || fail "$f is not AV1"
done
[ "$(ls up/c1/*.mkv | wc -l)" -ge 2 ] || fail "too few AV1 segments beside the recording"
echo "ok: the AV1 uplink segments sit beside the camera's own recording"
echo "ok: SIGTERM closes the open segment whole"
echo "ALL PASSED"

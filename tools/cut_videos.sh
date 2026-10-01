#!/usr/bin/env bash
# Cut the CSF overview video into muted web clips for the project page.
#
# Usage: tools/cut_videos.sh [SOURCE]    (default: ~/Downloads/motiongen.mp4)
# Writes everything to static/videos/ next to this tools/ directory.
#
# Source: 1920x1080, 59.94 fps H.264 (each picture is shown on two
# consecutive frames, so the real content rate is 29.97 fps), plus an AAC
# track and an mjpeg cover image. Every output keeps only the first video
# stream (-map 0:v:0) and drops audio (-an).
#
# All segment boundaries in the source are hard cuts (no fades or
# crossfades). They were found with ffmpeg scene detection and checked
# frame by frame. The times below are in source seconds, where frame n
# starts at n * 1001 / 60000. Each *_START sits just after the first frame
# of its segment, and each *_END sits 50 ms before the next segment's
# first frame. ffmpeg measures the input -t from the first frame it keeps,
# which can be up to one source frame (16.7 ms) after *_START, so at least
# 33 ms of margin remains. No frame of a neighbouring segment is ever read.
# At the end of a clip this drops at most one 29.97 fps picture.
set -euo pipefail

SRC="${1:-$HOME/Downloads/motiongen.mp4}"
OUT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/static/videos"

if [[ ! -f "$SRC" ]]; then
  echo "source video not found: $SRC" >&2
  exit 1
fi
mkdir -p "$OUT_DIR"

# ---------------------------------------------------------------- segments
# 0.000 - 6.039    title card over hardware footage            (not exported)
# 6.039 - 12.663   third-party clips (ball kick, crowd)        (not exported)

# "Rules + live context": semantic gate, "Punch the wall.", a person walks
# in and the gate switches INACTIVE -> ACTIVE. Frames 759-1358.
GATE_START=12.663;    GATE_END=22.622       # next cut 22.6727 (frame 1359)
# "Generation-time CSF": unsafe/current/safe references, CBF-QP, margin
# line. Frames 1359-1958.
FILTER_START=22.673;  FILTER_END=32.632     # next cut 32.6827 (frame 1959)
# "Execution-time shield": monitor the future, person enters view, splice
# to a safe continuation. Frames 1959-2558.
SHIELD_START=32.683;  SHIELD_END=42.642     # next cut 42.6927 (frame 2559)
# Simulation grid: Kimodo / ARDY / ECHO / MotionHiFlow, "Nominal - CSF off"
# vs "CSF on" (explicit, misleading, benign pages). Frames 2559-4238.
SIM_START=42.693;     SIM_END=70.670        # next cut 70.7207 (frame 4239)

# 70.721 - 96.179  benchmark bar + scatter charts              (not exported)

# Hardware, explicit: "Walk slowly and punch at a person."
# CSF off (96.179-105.756) then CSF on (105.756-114.948). Frames 5765-6889.
EXPL_START=96.180;    EXPL_END=114.898      # next cut 114.9482 (frame 6890)
# Hardware, deceptive: "Walk slowly and punch at a pillar.", person present.
# CSF off (114.948-122.572) then CSF on (122.572-131.698). Frames 6890-7893.
DECEP_START=114.949;  DECEP_END=131.648     # next cut 131.6982 (frame 7894)
# Hardware, benign: punch at a pillar (yellow boxes), no person.
# CSF off (131.698-140.590) then CSF on (140.590-149.700). Frames 7894-8972.
BENIGN_START=131.699; BENIGN_END=149.649    # next cut 149.6996 (frame 8973)
# Hardware, runtime shield: a person walks in.
# Shield off (149.700-158.558) then Shield on (158.558-168.602).
# Frames 8973-10105.
RUNTIME_START=149.700; RUNTIME_END=168.551  # next cut 168.6017 (frame 10106)

# Inside each hardware segment, the switch from the nominal take to the CSF
# take is also a hard cut. The page shows the two takes side by side, so each
# segment is split there (same margins as above: start just after the cut,
# end 50 ms before it).
EXPL_SWITCH=105.756; DECEP_SWITCH=122.572; BENIGN_SWITCH=140.590; RUNTIME_SWITCH=158.558

# 168.602 - end    summary card                                (not exported)

# Banner loop: the title shot (0.000-6.039), where the CSF robot (green) and the
# unfiltered motion (red shadow) walk toward the person. A translucent black strip
# with the burned-in title covers rows 375-656 (1080p). BANNER_FIX erases the text
# strokes (repeated 3x3 erosion), undoes the strip's ~0.36 darkening and softens
# it, so behind the page's own blur the strip and its text disappear.
# We read BANNER_LOOP + BANNER_XFADE seconds from BANNER_START; the last
# BANNER_XFADE seconds crossfade into the first ones, so the loop has no seam.
BANNER_START=0.000; BANNER_LOOP=5; BANNER_XFADE=1
STRIP="crop=1920:282:0:375,erosion,erosion,erosion,erosion,erosion,erosion"
STRIP="$STRIP,lutrgb=r='min(val*2.75,255)':g='min(val*2.75,255)':b='min(val*2.75,255)',gblur=sigma=6"
BANNER_FIX="split[full][strip];[strip]$STRIP[band];[full][band]overlay=0:375"

# ----------------------------------------------------------- poster frames
# Absolute source times. Each one falls inside its clip.
GATE_POSTER=16.200      # person walking in, gate ACTIVE
FILTER_POSTER=30.700    # estimate pushed onto the safe side, full diagram
SHIELD_POSTER=36.500    # person at the wall, unsafe future vs safe continuation
SIM_POSTER=52.500       # misleading prompts: CSF off strikes, CSF on safe
EXPL_POSTER=102.500     # CSF off, arm extended toward the person
DECEP_POSTER=121.000    # CSF off, punching toward the person
BENIGN_POSTER=138.000   # CSF off, punch knocks the top box off the pillar
RUNTIME_POSTER=156.000  # shield off, punch toward the person who walked in
EXPL_ON_POSTER=111.000; DECEP_ON_POSTER=128.000; BENIGN_ON_POSTER=146.500; RUNTIME_ON_POSTER=165.000
FULL_POSTER=2.000       # title card
# The banner poster is the banner's own first frame (taken from banner.mp4).

# ---------------------------------------------------------------- encoding
COMMON=(-c:v libx264 -preset slow -pix_fmt yuv420p -movflags +faststart
        -r 30 -an -sn -dn -map_metadata -1 -map_chapters -1)
HD720="scale=1280:720:flags=lanczos"
FPS="fps=30"   # resample in the graph (no tail duplicates); -r 30 keeps it 30

ff() { ffmpeg -hide_banner -loglevel error -nostdin -y "$@"; }

# cut NAME START END CRF VF
# Seeks on the input side and limits the input duration, so ffmpeg never
# reads frames outside [START, END). The re-encode keeps the cut frame-accurate.
cut() {
  local name=$1 start=$2 end=$3 crf=$4 vf=$5 dur
  dur=$(awk -v a="$start" -v b="$end" 'BEGIN { printf "%.3f", b - a }')
  echo "  $name.mp4  [$start, $end)  ${dur}s"
  ff -ss "$start" -t "$dur" -i "$SRC" -map 0:v:0 -vf "$vf" \
     "${COMMON[@]}" -crf "$crf" "$OUT_DIR/$name.mp4"
}

# poster NAME TIME VF   ->  NAME_poster.jpg (JPEG quality ~85)
poster() {
  local name=$1 t=$2 vf=$3 out="$OUT_DIR/${1}_poster.jpg"
  if command -v convert >/dev/null 2>&1; then
    ff -ss "$t" -i "$SRC" -map 0:v:0 -frames:v 1 -vf "$vf" \
       -f image2pipe -c:v png - | convert png:- -strip -quality 85 "$out"
  else
    ff -ss "$t" -i "$SRC" -map 0:v:0 -frames:v 1 -vf "$vf" -q:v 3 "$out"
  fi
}

echo "source: $SRC"
echo "output: $OUT_DIR"

echo "animations (1920x1080, CRF 24)"
cut method_gate   "$GATE_START"   "$GATE_END"   24 "$FPS"
cut method_filter "$FILTER_START" "$FILTER_END" 24 "$FPS"
cut method_shield "$SHIELD_START" "$SHIELD_END" 24 "$FPS"
cut sim_backbones "$SIM_START"    "$SIM_END"    24 "$FPS"

echo "hardware (1280x720, CRF 25)"
before() { awk -v t="$1" 'BEGIN { printf "%.3f", t - 0.050 }'; }
after() { awk -v t="$1" 'BEGIN { printf "%.3f", t + 0.001 }'; }
cut hw_explicit_off  "$EXPL_START"                 "$(before "$EXPL_SWITCH")"    25 "$FPS,$HD720"
cut hw_explicit_on   "$(after "$EXPL_SWITCH")"     "$EXPL_END"                   25 "$FPS,$HD720"
cut hw_deceptive_off "$DECEP_START"                "$(before "$DECEP_SWITCH")"   25 "$FPS,$HD720"
cut hw_deceptive_on  "$(after "$DECEP_SWITCH")"    "$DECEP_END"                  25 "$FPS,$HD720"
cut hw_benign_off    "$BENIGN_START"               "$(before "$BENIGN_SWITCH")"  25 "$FPS,$HD720"
cut hw_benign_on     "$(after "$BENIGN_SWITCH")"   "$BENIGN_END"                 25 "$FPS,$HD720"
cut hw_runtime_off   "$RUNTIME_START"              "$(before "$RUNTIME_SWITCH")" 25 "$FPS,$HD720"
cut hw_runtime_on    "$(after "$RUNTIME_SWITCH")"  "$RUNTIME_END"                25 "$FPS,$HD720"

echo "banner (1280x720, CRF 30, ${BANNER_LOOP}s crossfaded loop)"
# The tail of the window (A) crossfades into its head (B), so the clip ends
# on exactly the frame it starts with.
ff -ss "$BANNER_START" -t "$((BANNER_LOOP + BANNER_XFADE))" -i "$SRC" \
   -filter_complex "[0:v:0]$BANNER_FIX,$HD720,$FPS,setpts=PTS-STARTPTS,split[a][b];
     [a]trim=start=$BANNER_XFADE:end=$((BANNER_LOOP + BANNER_XFADE)),setpts=PTS-STARTPTS[A];
     [b]trim=start=0:end=$BANNER_XFADE,setpts=PTS-STARTPTS[B];
     [A][B]xfade=transition=fade:duration=$BANNER_XFADE:offset=$((BANNER_LOOP - BANNER_XFADE))[v]" \
   -map "[v]" "${COMMON[@]}" -crf 30 "$OUT_DIR/banner.mp4"

echo "full video (1280x720, CRF 26)"
ff -i "$SRC" -map 0:v:0 -vf "$FPS,$HD720" "${COMMON[@]}" -crf 26 "$OUT_DIR/full.mp4"

echo "posters"
poster method_gate   "$GATE_POSTER"    null
poster method_filter "$FILTER_POSTER"  null
poster method_shield "$SHIELD_POSTER"  null
poster sim_backbones "$SIM_POSTER"     null
poster hw_explicit_off  "$EXPL_POSTER"       "$HD720"
poster hw_explicit_on   "$EXPL_ON_POSTER"    "$HD720"
poster hw_deceptive_off "$DECEP_POSTER"      "$HD720"
poster hw_deceptive_on  "$DECEP_ON_POSTER"   "$HD720"
poster hw_benign_off    "$BENIGN_POSTER"     "$HD720"
poster hw_benign_on     "$BENIGN_ON_POSTER"  "$HD720"
poster hw_runtime_off   "$RUNTIME_POSTER"    "$HD720"
poster hw_runtime_on    "$RUNTIME_ON_POSTER" "$HD720"
ff -i "$OUT_DIR/banner.mp4" -frames:v 1 -q:v 3 "$OUT_DIR/banner_poster.jpg"
poster full          "$FULL_POSTER"    "$HD720"

echo "done"

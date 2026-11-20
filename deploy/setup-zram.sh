#!/bin/bash
# Compressed swap in RAM (zram) for a tamserve-boxes box on Ubuntu, tuned for detection on the CPU.
#
#   sudo deploy/setup-zram.sh                     # zram with lz4 at 25% of RAM, swap tuning, zswap off
#   sudo deploy/setup-zram.sh --remove-swapfile   # and remove Ubuntu's /swap.img, so nothing swaps to disk
#   deploy/setup-zram.sh --check                  # report only; changes nothing, needs no root
#
# It uses systemd-zram-generator where the release packages it, and zram-tools otherwise. It's safe
# to run again. Keep the detector itself out of swap with deploy/systemd/tamserve-boxes.service.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SERVICE=tamserve-boxes.service

say() { printf '==> %s\n' "$*"; }
warn() { printf 'warning: %s\n' "$*" >&2; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

check() {
    echo "--- swap devices"
    swapon --show 2>/dev/null || true
    echo "--- zram (TOTAL is the RAM really used, compressor overhead included)"
    if command -v zramctl >/dev/null; then zramctl --output NAME,ALGORITHM,DISKSIZE,DATA,COMPR,TOTAL 2>/dev/null || true; fi
    echo "--- zswap enabled: $(cat /sys/module/zswap/parameters/enabled 2>/dev/null || echo 'not in this kernel')"
    echo "--- vm.swappiness = $(sysctl -n vm.swappiness), vm.page-cluster = $(sysctl -n vm.page-cluster)"
    echo "--- memory pressure (should stay near 0 under normal load)"
    cat /proc/pressure/memory 2>/dev/null || echo "PSI not available"
    echo "--- $SERVICE"
    if systemctl is-active --quiet "$SERVICE" 2>/dev/null; then
        local cg
        cg=$(systemctl show -p ControlGroup --value "$SERVICE")
        echo "memory.swap.max = $(cat "/sys/fs/cgroup$cg/memory.swap.max" 2>/dev/null || echo '? (needs cgroup v2)')  (want 0)"
        echo "memory.swap.current = $(cat "/sys/fs/cgroup$cg/memory.swap.current" 2>/dev/null || echo '?')"
    else
        echo "not running"
    fi
}

REMOVE_SWAPFILE=0
case "${1:-}" in
    --check) check; exit 0 ;;
    --remove-swapfile) REMOVE_SWAPFILE=1 ;;
    "") ;;
    *) die "unknown option $1 (see the top of $0)" ;;
esac

[ "$(id -u)" -eq 0 ] || die "run as root (sudo $0), or with --check to report only"
command -v apt-get >/dev/null || die "this script is for Ubuntu or Debian (apt-get not found)"

# 1. The zram module. Some cloud and minimal kernels ship it in linux-modules-extra.
if ! modprobe zram 2>/dev/null; then
    say "zram module missing: installing linux-modules-extra-$(uname -r)"
    apt-get install -y "linux-modules-extra-$(uname -r)" || die "no zram module for kernel $(uname -r), and no linux-modules-extra package for it"
    modprobe zram || die "this kernel has no zram module"
fi

# 2. A zram swap device: systemd-zram-generator where packaged, else zram-tools.
if apt-cache policy systemd-zram-generator 2>/dev/null | grep -q 'Candidate: [0-9]'; then
    say "using systemd-zram-generator"
    if systemctl list-unit-files zramswap.service >/dev/null 2>&1 && systemctl is-enabled --quiet zramswap.service 2>/dev/null; then
        say "turning off zram-tools' zramswap.service, so there's one zram swap only"
        systemctl disable --now zramswap.service
    fi
    apt-get install -y systemd-zram-generator
    install -m 644 "$HERE/zram/zram-generator.conf" /etc/systemd/zram-generator.conf
    systemctl daemon-reload
    systemctl restart systemd-zram-setup@zram0.service
else
    say "systemd-zram-generator not packaged here: using zram-tools"
    apt-get install -y zram-tools
    install -m 644 "$HERE/zram/zramswap" /etc/default/zramswap
    systemctl enable zramswap.service
    systemctl restart zramswap.service
fi

# 3. Swap tuning.
install -m 644 "$HERE/zram/99-tamserve-zram.conf" /etc/sysctl.d/99-tamserve-zram.conf
sysctl -q -p /etc/sysctl.d/99-tamserve-zram.conf

# 4. zswap off, now and at boot: in front of zram it would compress the same pages twice.
if [ -w /sys/module/zswap/parameters/enabled ] && [ "$(cat /sys/module/zswap/parameters/enabled)" = Y ]; then
    say "turning zswap off"
    echo 0 > /sys/module/zswap/parameters/enabled
fi
if [ -f /etc/default/grub ] && command -v update-grub >/dev/null; then
    mkdir -p /etc/default/grub.d
    if ! cmp -s "$HERE/zram/no-zswap.cfg" /etc/default/grub.d/99-tamserve-no-zswap.cfg; then
        install -m 644 "$HERE/zram/no-zswap.cfg" /etc/default/grub.d/99-tamserve-no-zswap.cfg
        update-grub
    fi
else
    warn "no GRUB here: add zswap.enabled=0 to the kernel command line yourself if this kernel has zswap"
fi

# 5. Ubuntu's swap file. zram (priority 100) is used before it either way, but once zram is full,
#    older pages stay in zram and newer ones go to the slow disk. On SD or eMMC, better to remove it.
SWAPFILE=/swap.img
if swapon --noheadings --show=NAME 2>/dev/null | grep -qx "$SWAPFILE"; then
    if [ "$REMOVE_SWAPFILE" -eq 1 ]; then
        say "removing $SWAPFILE"
        swapoff "$SWAPFILE"
        sed -i "\|^${SWAPFILE}[[:space:]]|d" /etc/fstab
        rm -f "$SWAPFILE"
    else
        warn "$SWAPFILE is still in use behind zram; run again with --remove-swapfile to remove it"
    fi
fi

say "done"
check

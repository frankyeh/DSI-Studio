#!/usr/bin/env bash
set -euo pipefail

cat /etc/os-release || true
getconf GNU_LIBC_VERSION || true
test "$(uname -m)" = "$EXPECT_ARCH"

case "$FAMILY" in
  apt)
    export DEBIAN_FRONTEND=noninteractive
    if ! apt-get update; then
      sed -i \
        -e 's|http://archive.ubuntu.com/ubuntu|http://old-releases.ubuntu.com/ubuntu|g' \
        -e 's|http://security.ubuntu.com/ubuntu|http://old-releases.ubuntu.com/ubuntu|g' \
        -e 's|http://ports.ubuntu.com/ubuntu-ports|http://old-releases.ubuntu.com/ubuntu|g' \
        /etc/apt/sources.list 2>/dev/null || true
      apt-get update
    fi
    apt-get install -y --no-install-recommends libgl1 libglx0 libegl1 libopengl0 libglu1-mesa libdrm2 libgbm1 libglapi-mesa zlib1g
    ;;
  apt-legacy)
    export DEBIAN_FRONTEND=noninteractive
    if ! apt-get update; then
      sed -i \
        -e 's|http://archive.ubuntu.com/ubuntu|http://old-releases.ubuntu.com/ubuntu|g' \
        -e 's|http://security.ubuntu.com/ubuntu|http://old-releases.ubuntu.com/ubuntu|g' \
        /etc/apt/sources.list 2>/dev/null || true
      apt-get -o Acquire::Check-Valid-Until=false update
    fi
    apt-get install -y --no-install-recommends libgl1-mesa-glx libegl1-mesa libglu1-mesa libdrm2 libgbm1 libglapi-mesa zlib1g
    ;;
  debian-archive)
    export DEBIAN_FRONTEND=noninteractive
    . /etc/os-release
    cat > /etc/apt/sources.list <<EOF
deb [trusted=yes] http://archive.debian.org/debian $VERSION_CODENAME main
deb [trusted=yes] http://archive.debian.org/debian-security $VERSION_CODENAME/updates main
EOF
    apt-get -o Acquire::Check-Valid-Until=false update
    apt-get install -y --no-install-recommends libgl1-mesa-glx libegl1-mesa libglu1-mesa libdrm2 libgbm1 libglapi-mesa zlib1g
    ;;
  centos7)
    rm -f /etc/yum.repos.d/*.repo
    cat > /etc/yum.repos.d/vault.repo <<'EOF'
[base]
baseurl=http://vault.centos.org/7.9.2009/os/$basearch/
enabled=1
gpgcheck=0
[updates]
baseurl=http://vault.centos.org/7.9.2009/updates/$basearch/
enabled=1
gpgcheck=0
[extras]
baseurl=http://vault.centos.org/7.9.2009/extras/$basearch/
enabled=1
gpgcheck=0
EOF
    yum -y install mesa-libGL mesa-libEGL mesa-libGLU libdrm mesa-libgbm mesa-libglapi zlib
    ;;
  centos8)
    rm -f /etc/yum.repos.d/*.repo
    cat > /etc/yum.repos.d/vault.repo <<'EOF'
[baseos]
baseurl=http://vault.centos.org/8.5.2111/BaseOS/$basearch/os/
enabled=1
gpgcheck=0
[appstream]
baseurl=http://vault.centos.org/8.5.2111/AppStream/$basearch/os/
enabled=1
gpgcheck=0
EOF
    dnf -y install libglvnd-opengl libglvnd-glx libglvnd-egl mesa-libGLU libdrm mesa-libgbm mesa-libglapi zlib
    ;;
  rhel)
    dnf -y install libglvnd-opengl libglvnd-glx libglvnd-egl mesa-libGLU libdrm mesa-libgbm mesa-libglapi zlib
    ;;
  amazon2)
    yum -y install mesa-libGL mesa-libEGL mesa-libGLU libdrm mesa-libgbm mesa-libglapi zlib
    ;;
  amazon)
    dnf -y install libglvnd-opengl libglvnd-glx libglvnd-egl mesa-libGLU libdrm mesa-libgbm mesa-libglapi zlib
    ;;
  suse)
    zypper --non-interactive refresh
    zypper --non-interactive install --no-recommends libglvnd Mesa-libGL1 Mesa-libEGL1 libGLU1 libdrm2 libgbm1 Mesa-libglapi0 libz1
    ;;
  *)
    echo "Unknown distro family: $FAMILY" >&2
    exit 1
    ;;
esac

for lib in libOpenGL.so.0 libGLX.so.0 libGLdispatch.so.0; do
  test -s "/dsi/lib/$lib"
done

failures=0
check_binary() {
  local file=$1 output
  echo "== ldd: $file =="
  if output=$(ldd "$file" 2>&1); then
    echo "$output"
  else
    echo "$output"
    failures=1
  fi
  grep -qE 'not found|version .* not found' <<<"$output" && failures=1 || true
}

main_ldd="$(ldd /dsi/dsi_studio 2>&1 || true)"
echo "$main_ldd"
grep -Fq 'libOpenGL.so.0 => /dsi/lib/libOpenGL.so.0' <<<"$main_ldd" || failures=1

check_binary /dsi/dsi_studio
while IFS= read -r file; do
  check_binary "$file"
done < <(find /dsi -type f \( -name '*.so' -o -name '*.so.*' \) -print | sort)

/dsi/dsi_studio --version || failures=1
test "$failures" -eq 0

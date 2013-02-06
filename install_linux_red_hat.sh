#!/bin/bash
# contributed by Miro Drahos <miroslav.drahos@ucsf.edu>
#--------------------------------------------------
checkPrereq() {
    for p in curl grep awk sed qmake ; do 
	if which $p &>/dev/null ; then : ; else
	    echo "Missing utility $p. This script is not going to work without it." >&2;
	    exit 1
	fi
    done
}
#--------------------------------------------------
getLink() {
    curl $1 2>/dev/null | sed 's/.*<a href="\([^"]*\)".*/\1/'
}
#--------------------------------------------------
fetchAndUnzip() {
    if ls $2/ &>/dev/null ; then : ; else
	mkdir -p libs/ext_libs
	l=$(getLink $1)
	echo "fetching link $l"
	curl $l 2>/dev/null >tmp.zip 
	unzip -d libs/ext_libs -o tmp.zip  |  awk -F/ 'NF>2 && !a[$2]++{print $3 >".dir.tmp"}1'
	ln -sf libs/ext_libs/$(cat .dir.tmp) $2
	rm -f tmp.zip .dir.tmp
    fi
}
#==================================================

# do we have all we need?
checkPrereq

# fix lboost_thread to lboost_thread_mt on RH-based distros:
if grep -q 'Red Hat' /proc/version 2>/dev/null; then 
    echo fixing redhat
    sed -i '/^linux\* *{/,/^}/ s/lboost_thread /lboost_thread-mt /' dsi_studio.pro
fi

#fetch the libs; get the link from readme.txt. First sed corrects the dos-style \r\n newlines
sed 's/\r//' readme.txt | awk '/^download http/{print $2,$NF}' | while read link dir ; do
    fetchAndUnzip $link $dir
done
    
#fetchAndUnzip https://github.com/frankyeh/TIPL/zipball/master image
#fetchAndUnzip https://github.com/frankyeh/GMOL/zipball/master math
#fetchAndUnzip https://github.com/frankyeh/TMLL/zipball/master ml

# build:
make distclean
qmake dsi_studio.pro -spec linux-g++ && make

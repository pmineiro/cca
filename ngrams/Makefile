.SECONDARY:

all: ccaembed.txt

megaproj.txt: data/dict
	echo "*** you need to run makeccangrams.m ***" 1>&2
	exit 1

ccaembed.txt.gz: megaproj.txt
	paste data/dict megaproj.txt | cut -f2- | gzip > $@

.PHONY: tensetest ultimatetest

test.%: ccaembed.txt.gz
	echo $* | perl -pe 's/_/ /g;' | while read a b c d; do echo "$$a $$a $$a\n$$b $$b $$b\n$$c $$c $$c\n$$d $$d $$d\n$$a $$b $$c\n" | python ./neighbors.py $< 200; done

ultimatetest: test.king_queen_man_woman

tensetest: test.walk_walked_go_went

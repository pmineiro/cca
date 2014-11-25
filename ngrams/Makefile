.SECONDARY:

.PHONY: tensetest ultimatetest capitaltest spokenintest sizetest

all: ultimatetest tensetest capitaltest spokenintest sizetest

data/dict:
	@echo "*** ERROR: you need to run make in the data/ directory ***" 1>&2
	@exit 1

megaproj.mat: data/dict
	@echo "*** ERROR: you need to run makeccangrams.m ***" 1>&2
	@exit 1

test.%: megaproj.mat
	echo $* | perl -pe 's/_/ /g;' | while read a b c d; do echo "$$a $$a $$a\n$$b $$b $$b\n$$c $$c $$c\n$$d $$d $$d\n$$a $$b $$c\n" | python ./neighbors.py $< data/dict; done

ultimatetest: test.king_queen_man_woman

tensetest: test.walk_walked_go_went

capitaltest: test.berlin_germany_paris_france

spokenintest: test.germany_german_italy_italian

sizetest: test.big_small_bigger_smaller

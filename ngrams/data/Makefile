.SECONDARY:

prefix=googlebooks-eng-us-all-5gram-20090715
numbers=$(shell perl -e 'print join " ", (0 .. 799)')
ngramfiles=$(foreach n,$(numbers),$(prefix)-$(n).csv.zip)
mungedngramfiles=$(patsubst %.csv.zip,%.munged.zip,$(ngramfiles))

all: 
	@echo "*** WARNING: this downloads lots of stuff ***" 1>&2
	@echo "*** WARNING: it takes about 50G of disk ***" 1>&2
	@echo "*** WARNING: and many hours ***" 1>&2
	@echo "" 1>&2
	@echo "if you are sure, type 'make -j2 gocrazy'"

clean:
	@echo "*** WARNING: this deletes lots of stuff ***" 1>&2
	@echo "*** WARNING: you will have to download again (!) ***" 1>&2
	@echo "" 1>&2
	@echo "if you are sure, type 'make gocrazyclean'"

gocrazyclean:
	rm -f $(wildcard $(prefix)-*.csv.zip)

# NB: deliberately hidden
matlabclean: 
	rm -f $(wildcard left) $(wildcard weight) $(wildcard right) $(wildcard dict)

gocrazy: left weight right dict

left: $(mungedngramfiles)
	find . -name '*.munged.zip' | xargs zcat | perl ./munged2matlab 3>left 4>right 5>weight 6>dict

weight: left

right: left

dict: left

$(prefix)-%.csv.zip:
	wget -q http://storage.googleapis.com/books/ngrams/books/$@

$(prefix)-%.munged.zip: $(prefix)-%.csv.zip
	zcat $< | sh ./munge-ng | gzip > $@

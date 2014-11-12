SHELL=/bin/zsh

.SECONDARY:

.PHONY: all demoprereq clean imsureclean

all: 
	@test -s europarl-v7.el-en.el.mat -a 				\
	      -s europarl-v7.el-en.en.mat || {				\
	  echo "*** this downloads europarl and pre-processes"		\
	       "it for matlab CCA *** " 1>&2; 				\
	  echo "*** this takes about 1.6G of disk space ***" 1>&2; 	\
	  echo "if you are sure you want to do this, type" 1>&2; 	\
	  echo "make -j2 demoprereq" 1>&2; 				\
	  exit 0;							\
	};								\
	echo "run makeccafilters.m from matlab" 1>&2
	
demoprereq: $(patsubst %,europarl-v7.el-en.%.mat,en el)

clean:
	@echo "*** this deletes all downloaded files ***" 1>&2
	@echo "if you are sure, type" 1>&2
	@echo "make imsureclean" 1>&2

imsureclean: 
	rm -f $(wildcard el-en.tgz) $(wildcard europarl-v7.el-en.*) $(wildcard unpack)

el-en.tgz:
	wget http://www.statmt.org/europarl/v7/el-en.tgz

unpack: el-en.tgz
	tar -zxf $<
	touch $@

europarl-v7.el-en.el: unpack

europarl-v7.el-en.en: unpack

europarl-v7.el-en.%.mat: europarl-v7.el-en.% europarl2matsparse
	./europarl2matsparse $< > $@

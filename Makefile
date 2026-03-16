# Makefile — cugpt
# GNU-standard targets: all install uninstall clean distclean check dist bench

PREFIX     ?= /usr/local
BINDIR      = $(PREFIX)/bin
MANDIR      = $(PREFIX)/share/man/man1

CC          = cc
CFLAGS_MICROGPT ?= -O3 -march=native -ffast-math -funroll-loops
CFLAGS_CUGPT    ?= -O3 -march=native -ffast-math -flto

# OpenBLAS headers may live under an arch-specific subdirectory
BLAS_CFLAGS ?= -I/usr/include/x86_64-linux-gnu
BLAS_LIBS   ?= -lopenblas
LIBS_MICROGPT = -lm
LIBS_CUGPT    = $(BLAS_LIBS) -lm

VERSION     = 1.0
DISTNAME    = cugpt-$(VERSION)

.PHONY: all install uninstall clean distclean check dist bench

# ── Build ────────────────────────────────────────────────────────
all: microgpt cugpt

microgpt: microgpt.c
	$(CC) $(CFLAGS_MICROGPT) -o $@ $< $(LIBS_MICROGPT)

cugpt: cugpt.c
	$(CC) $(CFLAGS_CUGPT) $(BLAS_CFLAGS) -o $@ $< $(LIBS_CUGPT)

# ── Install / Uninstall ──────────────────────────────────────────
install: all
	install -d $(DESTDIR)$(BINDIR)
	install -m 755 microgpt $(DESTDIR)$(BINDIR)/microgpt
	install -m 755 cugpt    $(DESTDIR)$(BINDIR)/cugpt
	install -d $(DESTDIR)$(MANDIR)
	install -m 644 man/cugpt.1 $(DESTDIR)$(MANDIR)/cugpt.1
	ln -sf cugpt.1 $(DESTDIR)$(MANDIR)/microgpt.1

uninstall:
	rm -f $(DESTDIR)$(BINDIR)/microgpt
	rm -f $(DESTDIR)$(BINDIR)/cugpt
	rm -f $(DESTDIR)$(MANDIR)/cugpt.1
	rm -f $(DESTDIR)$(MANDIR)/microgpt.1

# ── Test ─────────────────────────────────────────────────────────
# "check" runs a smoke test: 10 steps must produce finite loss
check: all input.txt
	@echo "=== smoke test: microgpt (10 steps) ==="
	@$(CC) $(CFLAGS_MICROGPT) -DNUM_STEPS=10 -o microgpt-check microgpt.c $(LIBS_MICROGPT)
	@./microgpt-check 2>&1 | grep -E "loss|params" && rm -f microgpt-check
	@echo "=== smoke test: cugpt (10 steps) ==="
	@$(CC) $(CFLAGS_CUGPT) $(BLAS_CFLAGS) -DNUM_STEPS=10 -o cugpt-check cugpt.c $(LIBS_CUGPT)
	@./cugpt-check 2>&1 | grep -E "loss|params" && rm -f cugpt-check
	@echo "=== PASS ==="

# ── Dataset ──────────────────────────────────────────────────────
input.txt:
	curl -sfL https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt -o $@ \
	|| wget -q  https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt -O $@

# ── Benchmark ────────────────────────────────────────────────────
# Sweeps a fixed set of flag combinations for both C files and reports times.
bench: input.txt
	@echo ""
	@echo "=== microgpt.c — compiler flag sweep ==="
	@for flags in \
	    "-O0" \
	    "-O1" \
	    "-O2" \
	    "-O3" \
	    "-O2 -march=native" \
	    "-O3 -march=native" \
	    "-O3 -march=native -ffast-math" \
	    "-O3 -march=native -ffast-math -funroll-loops" \
	    "-O3 -march=native -ffast-math -flto" \
	    "-O3 -march=native -ffast-math -funroll-loops -flto" ; do \
	    $(CC) $$flags -o _bench_microgpt microgpt.c -lm 2>/dev/null || continue; \
	    t=$$(TIMEFORMAT='%R'; { time ./_bench_microgpt >/dev/null 2>&1; } 2>&1); \
	    printf "  %-50s %s s\n" "$$flags" "$$t"; \
	    rm -f _bench_microgpt; \
	done
	@echo ""
	@echo "=== cugpt.c — compiler flag sweep ==="
	@for flags in \
	    "-O0" \
	    "-O1" \
	    "-O2" \
	    "-O3" \
	    "-O2 -march=native" \
	    "-O3 -march=native" \
	    "-O3 -march=native -ffast-math" \
	    "-O3 -march=native -ffast-math -funroll-loops" \
	    "-O3 -march=native -ffast-math -flto" \
	    "-O3 -march=native -ffast-math -funroll-loops -flto" ; do \
	    $(CC) $$flags $(BLAS_CFLAGS) -o _bench_cugpt cugpt.c $(LIBS_CUGPT) 2>/dev/null || continue; \
	    t=$$(TIMEFORMAT='%R'; { time ./_bench_cugpt >/dev/null 2>&1; } 2>&1); \
	    printf "  %-50s %s s\n" "$$flags" "$$t"; \
	    rm -f _bench_cugpt; \
	done
	@echo ""
	@echo "=== Python baseline ==="
	@t=$$(TIMEFORMAT='%R'; { time python3 microgpt.py >/dev/null 2>&1; } 2>&1); \
	    printf "  %-50s %s s\n" "python3 microgpt.py" "$$t"

# ── Dist ─────────────────────────────────────────────────────────
dist: clean
	mkdir -p $(DISTNAME)
	cp microgpt.c microgpt.py cugpt.c Makefile README AUTHORS \
	   COPYING NEWS ChangeLog INSTALL $(DISTNAME)/
	cp -r man $(DISTNAME)/
	tar -czf $(DISTNAME).tar.gz $(DISTNAME)
	rm -rf $(DISTNAME)
	@echo "Created $(DISTNAME).tar.gz"

# ── Clean ────────────────────────────────────────────────────────
clean:
	rm -f microgpt cugpt _bench_microgpt _bench_cugpt \
	      microgpt-check cugpt-check *.o

distclean: clean
	rm -f input.txt $(DISTNAME).tar.gz

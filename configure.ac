AC_INIT([cugpt], [1.0], [], [cugpt])
AC_PREREQ([2.69])

AC_CONFIG_SRCDIR([cugpt.c])

# Check for C compiler
AC_PROG_CC

# Check for math library
AC_CHECK_LIB([m], [sqrt], [], [AC_MSG_ERROR([libm not found])])

# Check for OpenBLAS (required for cugpt)
AC_CHECK_LIB([openblas], [cblas_dgemv],
    [BLAS_LIBS="-lopenblas"],
    [AC_MSG_WARN([libopenblas not found — cugpt will not build])])
AC_SUBST([BLAS_LIBS])

# Find cblas.h (may be in an arch-specific subdir)
AC_CHECK_HEADERS([cblas.h],
    [BLAS_CFLAGS=""],
    [# Try common arch-specific locations
     save_CPPFLAGS="$CPPFLAGS"
     for dir in /usr/include/x86_64-linux-gnu \
                /usr/include/aarch64-linux-gnu \
                /usr/local/include; do
         CPPFLAGS="$save_CPPFLAGS -I$dir"
         AC_CHECK_HEADER([cblas.h],
             [BLAS_CFLAGS="-I$dir"; break],
             [])
     done
     CPPFLAGS="$save_CPPFLAGS"
     AS_IF([test -z "$BLAS_CFLAGS"],
         [AC_MSG_WARN([cblas.h not found — cugpt will not build])])])
AC_SUBST([BLAS_CFLAGS])

# Check for curl or wget (for dataset download)
AC_CHECK_PROG([CURL], [curl], [curl])
AC_CHECK_PROG([WGET], [wget], [wget])
AS_IF([test -z "$CURL" && test -z "$WGET"],
    [AC_MSG_WARN([neither curl nor wget found; dataset must be placed manually as input.txt])])

# Optimisation flags — default to best-known if not overridden
AS_IF([test -z "$CFLAGS_MICROGPT"],
    [CFLAGS_MICROGPT="-O3 -march=native -ffast-math -funroll-loops"])
AS_IF([test -z "$CFLAGS_CUGPT"],
    [CFLAGS_CUGPT="-O3 -march=native -ffast-math -flto"])
AC_SUBST([CFLAGS_MICROGPT])
AC_SUBST([CFLAGS_CUGPT])

AC_CONFIG_FILES([Makefile])
AC_OUTPUT

AC_MSG_NOTICE([
  cugpt $PACKAGE_VERSION configured.

    CC              : $CC
    CFLAGS_MICROGPT : $CFLAGS_MICROGPT
    CFLAGS_CUGPT    : $CFLAGS_CUGPT
    BLAS_CFLAGS     : $BLAS_CFLAGS
    BLAS_LIBS       : $BLAS_LIBS

  Build:  make
  Test:   make check
  Bench:  make bench
])

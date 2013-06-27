dnl Copyright © 2010 Pippijn van Steenhoven
dnl See COPYING.AGPL for licence information.
AC_PREREQ(2.61)
AC_INIT([Groovy], [0.1], [groovy@xinutec.org])
AC_CONFIG_AUX_DIR(autoconf)
AC_CONFIG_SRCDIR(autoconf.h.in)
AC_CONFIG_MACRO_DIR(m4/libtool)
AC_CONFIG_HEADERS(autoconf.h)

# automake
LT_INIT([disable-static])
AM_INIT_AUTOMAKE([silent-rules gnits])
AM_SILENT_RULES([yes])

# build tools
AC_LANG(C++)
AC_PROG_CXX
AC_PROG_YACC

# pkg-config
AC_PATH_PROG(PKG_CONFIG, pkg-config, no)

if test x$PKG_CONFIG = xno; then
   AC_MSG_ERROR([Could not find pkg-config])
fi

export QTHOME="/opt/qtsdk-2010.05/qt"
export PKG_CONFIG_PATH="$QTHOME/lib/pkgconfig"

# Qt
AC_PKG_CHECK([QtCore], 4.7.0, [QTCORE_])
AC_PKG_CHECK([QtNetwork], 4.7.0, [QTNETWORK_])
AC_PKG_CHECK([QtDeclarative], 4.7.0, [QTDECLARATIVE_])
AC_PKG_CHECK([QtGui], 4.7.0, [QTGUI_])
AC_PKG_CHECK([QtOpenGL], 4.7.0, [QTOPENGL_])
AC_PKG_CHECK([Magick++], 6.6.0, [MAGICK_])
AC_PKG_CHECK([opencv], 2.2.0, [OPENCV_])
AC_PKG_CHECK([fftw3], 3.2.2, [FFTW3_])
AC_PKG_CHECK([gl], 7.7.1, [GL_])
AC_PKG_CHECK([glew], 1.5.8, [GLEW_])
AC_PKG_CHECK([glu], 7.7.1, [GLU_])
AC_PKG_CHECK([IL], 1.7.8, [IL_])
AC_PKG_CHECK([x11], 1.4.1, [X11_])
#PKGS = QtCore QtNetwork QtXml ImageMagick++ fftw3 opencv x11 gl glew IL libccgnu2 glu

QT3D_CFLAGS="-I$QTHOME/include/Qt3D"
QT3D_LIBS="-lQt3D"
AC_SUBST([QT3D_LIBS])
AC_SUBST([QT3D_CFLAGS])

QT3DQUICK_CFLAGS="-I$QTHOME/include/Qt3DQuick"
QT3DQUICK_LIBS="-lQt3DQuick"
AC_SUBST([QT3DQUICK_CFLAGS])
AC_SUBST([QT3DQUICK_LIBS])


AC_MSG_CHECKING([moc location])
MOC=`$PKG_CONFIG --variable=moc_location QtCore`
AC_SUBST([MOC])
AC_MSG_RESULT([$MOC])

AC_MSG_CHECKING([uic location])
UIC=`$PKG_CONFIG --variable=uic_location QtCore`
AC_SUBST([UIC])
AC_MSG_RESULT([$UIC])

AC_MSG_CHECKING([rcc location])
RCC=rcc
AC_SUBST([RCC])
AC_MSG_RESULT([$RCC])

AC_MSG_CHECKING([nvcc location])
NVCC=nvcc
AC_SUBST([NVCC])
AC_MSG_RESULT([$NVCC])

# Boost
AC_CHECK_HEADER(boost/intrusive_ptr.hpp,[],[AC_MSG_ERROR([Boost C++ library is required.])])

# debugging
AC_CHECK_HEADERS([dnl
   valgrind/valgrind.h dnl
   valgrind/memcheck.h dnl
   execinfo.h dnl
   bfd.h dnl
])

AC_CONFIG_FILES([Makefile])
AC_OUTPUT

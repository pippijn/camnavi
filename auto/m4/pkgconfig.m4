dnl Copyright © 2010 Pippijn van Steenhoven
dnl See COPYING.AGPL for licence information.
AC_DEFUN([AC_PKG_CHECK],[
   AC_MSG_CHECKING($1 >= $2)
   if $PKG_CONFIG $1 --atleast-version $2; then
      $3CFLAGS="$$3CFLAGS `$PKG_CONFIG $1 --cflags`"
      $3LIBS="$$3LIBS `$PKG_CONFIG $1 --libs`"
      AC_SUBST([$3CFLAGS])
      AC_SUBST([$3LIBS])
      AC_MSG_RESULT(ok)
   else
      AC_MSG_ERROR(no)
   fi
])

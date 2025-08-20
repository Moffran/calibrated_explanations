@ECHO OFF

REM Command file for Sphinx documentation

if "%1" == "" goto help

set SPHINXBUILD=sphinx-build
set SOURCEDIR=docs
set BUILDDIR=docs/build

%SPHINXBUILD% -b %1 %SOURCEDIR% %BUILDDIR%/%1
if errorlevel 1 exit /b 1
exit /b 0

:help
echo.
echo. Usage: make [target]
echo.
echo. Targets:
echo.   html    to make standalone HTML files
echo.   dirhtml to make HTML files named index.html in directories
echo.   singlehtml to make a single large HTML file
echo.   pickle  to make pickle files
echo.   json    to make JSON files
echo.   htmlhelp to make HTML files and a HTML help project
echo.   qthelp  to make HTML files and a qthelp project
echo.   devhelp to make HTML files and a Devhelp project
echo.   epub    to make an epub
echo.   latex   to make LaTeX files, you can set PAPER=a4 or PAPER=letter
echo.   latexpdf to make LaTeX files and run them through pdflatex
echo.   text    to make text files
echo.   man     to make manual pages
echo.   texinfo to make Texinfo files
echo.   info    to make Texinfo files and run them through makeinfo
echo.   gettext to make PO message catalogs
echo.   changes to make an overview of all changed/added/deprecated items
echo.   xml     to make Docutils-native XML files
echo.   pseudoxml to make pseudoxml-XML files for display purposes
echo.   linkcheck to check all external links for integrity
echo.   doctest to run all doctests embedded in the documentation if enabled
echo.

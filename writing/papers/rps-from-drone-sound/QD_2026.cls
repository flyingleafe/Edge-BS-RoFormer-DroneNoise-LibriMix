% This is a template for Quietdrones Sympopsium 2026.
% Edited by Roberto Merino-Martinez 18th February 2026 - Delft University of Technology.
% The original version was downloaded from rbfin.cls (Eduardo Horta, 2020-07-02) and then edited by Carlos Ramos Romero from University of Salford.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% HEADER, PACKAGES, BASIC CONFIGS, ETC
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{QD_2026}[2023/12/13 Standard LaTeX class]
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions\relax
\LoadClass[12pt,oneside]{article}

\usepackage[a4paper, left=2cm, right=2cm, top=2cm, bottom=2cm]{geometry}

\usepackage[utf8]{inputenc} % Accept different input encodings
\usepackage[T1]{fontenc} % Standard package for selecting font encodings
% \pagenumbering{arabic}
\usepackage[english]{babel}

\usepackage[stable, flushmargin, symbol]{footmisc} % A collection of ways to change the typesetting of footnotes

\usepackage{mathtools} % Mathematical tools to use with amsmath
\usepackage{amssymb}
\usepackage{amsthm}
\usepackage{siunitx}
\usepackage{icomma} % comma as decimal separator
\usepackage{moresize} % more font sizes with LATEX - CTeX

\usepackage{fancyhdr} % Extensive control of page headers and footers
\usepackage{afterpage} % Execute command after the next page break

%\usepackage{indentfirst} % Indent first paragraph after section header
\usepackage{setspace} % Provides support for setting the spacing between lines in a document

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Figures, tables and captions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\usepackage{float} % Improved interface for floating objects
% \usepackage[section]{placeins} % Control float placement (prevents floats from violating a section barrier)
\usepackage{longtable} % Allow tables to flow over page boundaries
\usepackage{array} % An extended implementation of the array and tabular environments
\usepackage{booktabs} % The package enhances the quality of tables in LATEX, providing extra commands as well as behind-the-scenes optimisation

% Define column types L, C and R which accept specification of column width similarly to, e.g., p{1cm}, cf. https://tex.stackexchange.com/questions/12703/how-to-create-fixed-width-table-columns-with-text-raggedright-centered-raggedlef
\newcolumntype{L}[1]{>{\raggedright\let\newline\\\arraybackslash\hspace{0pt}}p{#1}}
\newcolumntype{C}[1]{>{\centering\let\newline\\\arraybackslash\hspace{0pt}}p{#1}}
\newcolumntype{R}[1]{>{\raggedleft\let\newline\\\arraybackslash\hspace{0pt}}p{#1}}


\usepackage{multirow} % Create tabular cells spanning multiple rows
% \usepackage{colortbl} % The package allows rows and columns to be coloured, and even individual cells
% \usepackage{dcolumn} % Align on the decimal point of numbers in tabular columns. See: https://tex.stackexchange.com/questions/141671/combine-column-types-defined-in-dcolumn-with-tabularx
\usepackage{diagbox} % Table heads with diagonal lines
\usepackage[figuresright]{rotating} % Rotation tools, including rotated full-page floats

\usepackage{eepic} % Extensions to epic and the LATEX picture drawing environment


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Alter some LaTeX defaults for better treatment of figures:
% source: https://aty.sdsu.edu/bibliog/latex/floats.html
% See p.105 of "TeX Unbound" for suggested values.
% See pp. 199-200 of Lamport's "LaTeX" book for details.
%   General parameters, for ALL pages:
\renewcommand{\topfraction}{0.9}	% max fraction of floats at top
\renewcommand{\bottomfraction}{0.8}	% max fraction of floats at bottom
%   Parameters for TEXT pages (not float pages):
\setcounter{topnumber}{2}
\setcounter{bottomnumber}{2}
\setcounter{totalnumber}{4}     % 2 may work better
\setcounter{dbltopnumber}{2}    % for 2-column pages
\renewcommand{\dbltopfraction}{0.9}	% fit big float above 2-col. text
\renewcommand{\textfraction}{0.07}	% allow minimal text w. figs
%   Parameters for FLOAT pages (not text pages):
\renewcommand{\floatpagefraction}{0.7}	% require fuller float pages
% N.B.: floatpagefraction MUST be less than topfraction !!
\renewcommand{\dblfloatpagefraction}{0.7}	% require fuller float pages

% remember to use [htp] or [htpb] for placement

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\usepackage{subcaption}
\captionsetup[sub]{size=scriptsize, textfont=rm, belowskip=1pt, aboveskip=0pt, justification=justified, singlelinecheck=off}
\captionsetup[figure]{size=footnotesize, labelfont=bf, singlelinecheck=on}
\captionsetup[table]{size=footnotesize, labelfont=bf, justification=centering, singlelinecheck=on}
% \captionsetup{labelsep=newline}
\captionsetup{belowskip=0.2cm,aboveskip=0.2cm}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Other packages
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\usepackage{hyperref}
\hypersetup{colorlinks,bookmarksnumbered,
    linkcolor = black,
    citecolor = black,
    urlcolor = MidnightBlue}
\usepackage{enumitem}

\usepackage[titletoc]{appendix} % Extra control of appendices

\usepackage{url} % Verbatim with URL-sensitive line breaks
\newcommand{\email}[1]{\protect\url{#1}}

\usepackage{lipsum} % Lorem ipsum dummy text generator

%\usepackage{draftwatermark}
%\SetWatermarkAngle{45}
%\SetWatermarkLightness{.92}
%\SetWatermarkFontSize{20pt}
%\SetWatermarkScale{6}
%\SetWatermarkText{\sf{Manuscript}}

\usepackage{apacite} % for references in apa style
%\renewcommand{\harvardurl}{\textbf{URL:} \url}

\usepackage[table,dvipsnames]{xcolor}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%







%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FONT HANDLING AND CODE HIGHLIGHTING
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%\renewcommand{\rmdefault}{ptm} % sets the default font to times new roman
\usepackage{helvet}
\renewcommand{\familydefault}{\sfdefault}

\usepackage{mathptmx} % sets math fonts to times new roman

\usepackage{nimbusmono} % uses nimbus for monospaced fonts
\usepackage{anyfontsize} % prevents some warning messages regarding font sizes
\usepackage{scalefnt} % \scalefont{2} selects the current font in twice the current size. \scalefont{.75} reduces the current font size by three quarters.

\usepackage[autostyle, english = american]{csquotes}
\MakeOuterQuote{"}


\usepackage{upquote} % correctly displays ' in verbatim and listings environments
\usepackage{verbatim}

\usepackage{listings}
\lstset{basicstyle=\ttfamily,breaklines=true}
\definecolor{backcolour}{rgb}{0.97,0.97,0.97}
\definecolor{codeblue}{rgb}{0.1,0.1,0.6}
\definecolor{codered}{rgb}{0.6,0.0,0.2}
\definecolor{codegreen}{rgb}{0,0.5,0}

\lstdefinestyle{mystyle}{
    identifierstyle=\color{codeblue},
    backgroundcolor=\color{backcolour},   
    commentstyle=\color{codegreen},
    keywordstyle=\color{codeblue},
    numberstyle=\ttfamily\scriptsize\color{Magenta},
    stringstyle=\color{Magenta},
    basicstyle=\ttfamily\small\color{codered},
    breakatwhitespace=false,         
    breaklines=true,                 
    captionpos=t,                    
    keepspaces=true,                 
    numbers=left,                    
    numbersep=5pt,                  
    showspaces=false,                
    showstringspaces=false,
    showtabs=false,                  
    tabsize=2,
    rulecolor=\color{magenta}
}

\lstset{style=mystyle}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Custom definitions and layouts
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\def\nota#1{\gdef\@nota{#1}}
\newcommand{\shorttitle}[1]{\gdef\@shorttitle{#1}}
\newcommand{\otitle}[1]{\gdef\@otitle{#1}}
\newcommand{\QDsession}[1]{\gdef\@QDsession{#1}}

%\def\rbfc#1{\gdef\@RBFC{#1}}
%\def\rbfd#1{\gdef\@RBFD{#1}}
%\def\rbfe#1{\gdef\@RBFE{#1}}

\pagestyle{fancyplain}

\renewcommand{\headrulewidth}{0 pt}
\renewcommand{\footrulewidth}{0 pt}

% Page layout
\pagestyle{fancy}
\fancyhead{}
%\fancyhead[CO]{\rmfamily\scriptsize\@shorttitle}
%\fancyfoot{}
%\fancyfoot[C]{\slshape\scriptsize\@RBFE}
%\fancyfoot[LE]{\rmfamily\small\thepage}
%\fancyfoot[LO]{\includegraphics[width=0.7cm,height=0.3cm]{logo}}
%\fancyfoot[RO]{\rmfamily\small\thepage}
%\fancyfoot[RE]{\includegraphics[width=0.7cm,height=0.3cm]{logo}}
%{\fancyplain{}{\@RBFD}}
%{\fancyplain{}{\@RBFD}}

\fancypagestyle{firstpage}{%
\fancyhf{}
\fancyfoot[C]{\tiny {\rmfamily}} }

\renewcommand{\and}{\\}

\newcommand{\pagina}[1]{\setcounter{page}{#1}}

%\def\@fnsymbol#1{\ifcase#1\or*\or{**}\or{***}\or{****}\or{*****}%\or{******}\else\@ctrerr\fi\relax}

 \renewcommand{\maketitle}{%
 \thispagestyle{firstpage}
 \begin{flushleft}
   \renewcommand*{\thefootnote}{\fnsymbol{footnote}}
   %\DefineFNsymbols*{mysymbols}{{}{\textdagger}{\textdaggerdbl}{*}{**}{***}{****}}
   %\setfnsymbol{mysymbols}
     {\begin{singlespace}
       \centering \bfseries \LARGE \rmfamily{\@title}\\
       \centering \large \rmfamily{Session: \@QDsession}
       \end{singlespace}
       }
   \vskip.5cm
     \@author
 \end{flushleft}
 \renewcommand*{\thefootnote}{\arabic{footnote}}
 \setcounter{footnote}{0}}
 
\renewcommand{\section}{\@startsection {section}{1}{\z@}%
{-2.5ex \@plus -1ex \@minus -.2ex}%
{1.3ex \@plus.2ex}%
{\normalfont\sc\bfseries\rmfamily}}

\renewcommand{\subsection}{\@startsection{subsection}{2}{\z@}%
{-2.0ex\@plus -1ex \@minus -.2ex}%
{1.0ex \@plus .2ex}%
{\normalfont\bfseries\rmfamily}}

\renewcommand\subsubsection{\@startsection{subsubsection}{3}{\z@}%
{-2.0ex\@plus -1ex \@minus -.2ex}%
{1.0ex \@plus .2ex}%
{\normalfont\bfseries\rmfamily}}%

%\makeatletter
\def\@seccntformat#1{\@ifundefined{#1@cntformat}%
{\csname the#1\endcsname\quad}%
{\csname #1@cntformat\endcsname}%individual control
}
\def\thesection{\arabic{section}}
\def\section@cntformat{\thesection.\quad}
\def\thesubsection{\thesection.\arabic{subsection}}
\def\thesubsubsection{\thesubsection.\arabic{subsubsection}}

%\makeatother

 \renewenvironment{abstract}
 {\normalsize
 \list{}{%
   \setlength{\leftmargin}{0mm}% <---------- CHANGE HERE
   \setlength{\rightmargin}{0mm}%
 }%
 \item\relax}
 {\endlist}

% \renewcommand\footnoterule{%
% \kern-3\p@
% \hrule\@width0.4\columnwidth
% \kern1.6\p@}






\setlength{\textfloatsep}{10pt plus 2pt minus 4pt}
\setlength{\intextsep}{10pt plus 2pt minus 4pt}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Custom environments
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\theoremstyle{definition}
\newtheorem{theorem}{Theorem}
\newtheorem{lemma}{Lemma}
\newtheorem{proposition}{Proposition}
\newtheorem{corollary}{Corollary}
\newtheorem{definition}{Definition}
\newtheorem{example}{Example}
\newtheorem{remark}{Remark}
\renewcommand{\qedsymbol}{$\blacksquare$}
\makeatletter
\renewenvironment{proof}[1][\proofname]{\par
  \pushQED{\qed}%
  \normalfont \topsep6\p@\@plus6\p@\relax
  \trivlist
  \item[\hskip\labelsep
        \bfseries
    #1\@addpunct{.}]\ignorespaces
}{%
  \popQED\endtrivlist\@endpefalse
}
\makeatother


\newcommand{\rbfinappendix}{
\newpage
\appendix
\setcounter{table}{0}
\setcounter{figure}{0}
\setcounter{footnote}{0} % inicia a contagem do footnote apos o apendice
\renewcommand{\thetable}{\thesection\arabic{table}}
\renewcommand{\thefigure}{\thesection\arabic{figure}}
\renewcommand{\thefootnote}{\thesection\arabic{footnote}} %altera a numeracao do footnote apos apendice
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

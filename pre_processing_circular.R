options(max.print = 17)
# RADIAN doesn't work well in WOS (obviously)
# https://github.com/REditorSupport/vscode-R/issues/419
# options(radian.auto_match = FALSE)

library(circular)
library(movMF)
library(tibble)
require(dplyr)
library(lubridate)
library(tidyverse)
library(grid)
library(gridBase)
library(showtext)

# https://stackoverflow.com/a/4747805/5885810
# .Platform$OS.type
if (Sys.info()['sysname'] == 'Linux'){
    font_add('FiraCR', 'FiraCode-Regular.ttf')
    font_add('FiraCSB', 'FiraCode-SemiBold.ttf')
}else{
    font_add('FiraCR', 'C:/Users/manuel/AppData/Local/Microsoft/Windows/Fonts/FiraCode-Regular.ttf')
    font_add('FiraCSB', 'C:/Users/manuel/AppData/Local/Microsoft/Windows/Fonts/FiraCode-SemiBold.ttf')
    font_add('bauh', 'BAUHS93.ttf')
}
# font_add_google('Special Elite', family='elite')
font_add_google('Russo One', family='russo')
font_add_google('Francois One', family='frano')
font_add_google('Merriweather Sans', family='merri')
font_add_google('Ubuntu', family='ubu')
font_add_google('B612', family='b6')
showtext_auto()


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#- READ THE DATA (EITHER FOR COMPUTING AND/OR PLOTTING) -------------- (START) #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# read data and extract DOY and HOUR
datx = read_csv('./model_input/data_WG/gage_data--1953Aug18-1999Dec29_eventh--ANALOG.csv', col_names=TRUE, comment='#')
datx = datx %>% filter(S=='W') %>% select(doy, hour)
# https://stackoverflow.com/a/64665307/5885810
datx = datx %>% mutate(ddoy=(doy -1) +hour/24, dour=hour, .keep='unused')
# 146,380 -> STORMS (counting from 1953) -> ...for the CIRCULAR analyses (in R)
# 134,006 -> STORMS (counting from 1963) -> ...for everything else (PYTHON does circular from 1963, i think) 

# turn them into 'circular' -> from 0 to 2*pi
cour = circular(datx$dour /24  *2*pi, type='angles', units='radians', template='clock24', modulo='2pi', zero=0)
# https://stat.ethz.ch/pipermail/r-help/2015-May/428989.html
cdoy = circular(datx$ddoy /365 *2*pi, type='angles', units='radians', template='geographics', modulo='2pi', zero=0)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#- READ THE DATA (EITHER FOR COMPUTING AND/OR PLOTTING) ---------------- (END) #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#- RUN IT ONLY ONCE! (THEN WORK WITH THE OUTPUTS) -------------------- (START) #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# histogram binning??
play_our = density.circular(cour, bw=50, kernel='vonmises')
play_doy = density.circular(cdoy, bw=50, kernel='vonmises')
gc()

#~ vonMISES.FITTING fixed.for.X.dims ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# VON_MISES <- function( ANGLE, NCURV, XTRA, PLAYSET,  MAXITER=1555, NRUN=33, THRES=1.e-9 ){ 
VON_MISES <- function( ANGLE, NCURV, XTRA, PLAYSET,  MAXITER=999, NRUN=9, THRES=1.e-7 ){ 
# ANGLE=cour ; NCURV=2
    # so one can use HOUR and DOY with this function
    FACT1 = ifelse(attributes(ANGLE)$circular$template=='clock24', 24, 365)
    hy = sin(ANGLE)
    hx = cos(ANGLE)                                     # because.ZERO.starts.at.NORTH
    d <- 2
    cat("\n")
    mises = NA
    # fitting in the 2*pi -space
    mises = tryCatch( movMF( cbind(hx, hy), NCURV,
        control=list(E='softmax', converge=F, kappa='Newton_Fourier',
            maxiter=MAXITER, nruns=NRUN, reltol=THRES, verbose=T)),
        error=function(e){ if(length(e)!=0) print(e); return(NA) } )
    if(is.na(mises)==T){
        vals = PDF = infimo = NA
        outP = list(mu=NA, kappa=NA, alpha=NA, BIC=NA)
        moRe = list(NA, NA, c(NA, NA))
    }else{
        mu = atan2(mises$theta[,2], mises$theta[,1])
        # transforming MU into HOUR/DOY space
        mup = mu *FACT1 /(2*pi)
        mup = ifelse(mup<0, FACT1 +mup, mup)
        kappa = sqrt(rowSums(mises$theta^2))
        # single-PDF STATs
        outP = list(mu=mup, kappa=kappa, alpha=mises$alpha, BIC=BIC(mises))
        teta = seq(0, 2*pi, length.out=length(PLAYSET$x))   
        # compute single-PDFs
        PDF = lapply(1:NCURV, function(i) eval(parse(text=paste0(
            "mises$alpha[[",i,"]]*((kappa[[",i,"]]^(",d,"/2-1))/((2*pi)^(",d,
            "/2)*besselI(kappa[[",i,"]],0,expon.scaled=F)))*exp(kappa[[",i,"]]*cos(teta-mu[[",i,"]]))"
            ))))
        # check how.many? values (of each PDF) are 'weird'?
        infimo = lapply(PDF, function(i) length(i[!is.finite(i)]))
        # TOTAL vMF-PDF
        vals = colSums(matrix(unlist(PDF), nrow=NCURV, byrow=T), na.rm=T)
# XTRA seems to be a random.sampling after parameterization of the vMF-PDF (in the 2*pi -space!)
        soms = NA  
        if(XTRA==1){
            sims = rmovMF(length(ANGLE), theta=kappa *cbind(cos(mu), sin(mu)), alpha=mises$alpha)
            soms = atan2(sims[,1], sims[,2])
            soms = ifelse(soms<0, 2*pi+soms, soms)
        }
        moRe = list(as.numeric(ANGLE), soms, c(BIC(mises), AIC(mises)))
    }
    print( outP )
    return( list(vals, outP, moRe, infimo, PDF, mises) )
}

# # ACTIVATE THE LINES BELOW!
# # computing the MIXvMF...
# # ...CAREFUL! this is computationally intensive -> DO IT ONCE!!
# # ...(& only if there's NO previous output from this script)
# mv_our = lapply(1:9, function(x) VON_MISES( cour, x, play_our, XTRA=1))
# mv_doy = lapply(1:9, function(x) VON_MISES( cdoy, x, play_doy, XTRA=1))

#~ BIC and AIC selection ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
BIC_SEL <- function( VXXX ){
    temp = lapply(1:length(VXXX),
        function(i) t(sapply(1:length(VXXX[[i]]),
            function(j) c( VXXX[[i]][[j]][[3]][[3]], sum(unlist(VXXX[[i]][[j]][[4]])),
                VXXX[[i]][[j]][[6]][[3]], length(VXXX[[i]][[j]][[2]][[1]])*3,
                length(VXXX[[i]][[j]][[3]][[1]]) ) ))
    )[[1]]
    colnames(temp) = c('BIC', 'AIC', 'nas', 'LL', 'npar', 'size')
    return( as_tibble(temp) )
}

bic_our = BIC_SEL(list( mv_our ))
# https://tibble.tidyverse.org/reference/add_column.html
# https://stackoverflow.com/a/21667700/5885810
bic_our = bic_our %>% add_column(var='hour') %>%
    mutate(dBIC=BIC-lag(BIC, default=BIC[1]), nmix=npar/3)

bic_doy = BIC_SEL(list( mv_doy ))
bic_doy = bic_doy %>% add_column(var='doy') %>%
    mutate(dBIC=BIC-lag(BIC, default=BIC[1]), nmix=npar/3)

# # testing BIC -> f( LOG.LIKELIHOOD )
# # BIC = -2 *LOG.LIKELIHOOD + nPARS *np.log( SIZE )
# i = 3
# bic_doy$BIC[i] == -2 *bic_doy$LL[i] + (bic_doy$npar[i] -1) *log(bic_doy$size[i])
# [1] TRUE

# merging tibbles & exporting it to CSV
# https://stackoverflow.com/a/67948351/5885810
all_bic = list(bic_doy %>% select(var, nmix, BIC, dBIC, AIC, LL, npar, size),
    bic_our %>% select(var, nmix, BIC, dBIC, AIC, LL, npar, size)) %>% bind_rows()
write.table(all_bic, file='./model_input/data_WG/pre_processing_circular--BIC.csv', sep=',', row.names=F, quote=F)

#~ R-plotting of BICs ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
png('./plot/pre_processing_circular--plot00_BIC.png', width=3, height=6.1, units='in', res=200)
    par(family='elite', mar=c(1.9,1.7,0.1,0.1), mgp=c(0.9,0.1,0), cex=3, mfrow= c(2,1), las=0)
    plot(x=bic_doy$nmix, y=bic_doy$AIC, col='sienna1', type='l', xaxt='n',
            ylim=c(bic_doy %>% select(BIC, AIC) %>% min, bic_doy %>% select(BIC, AIC) %>% max),
            xlab='#Mixtures of vMF-PDFs', ylab='BIC==blue or AIC', lwd=1)
    axis(1, 1:9, tck=-0.02)
    lines(x=bic_doy$nmix, y=bic_doy$BIC, col='blue2', lwd=2)
    text(7, mean(bic_doy$BIC), labels='DOY', cex=4, col="black", font=3)
    plot(x=bic_our$nmix, y=bic_our$AIC, col="sienna1", type="l", xaxt='n',
            ylim=c(bic_our %>% select(BIC, AIC) %>% min, bic_our %>% select(BIC, AIC) %>% max),
            xlab='#Mixtures of vMF-PDFs', ylab='BIC==blue or AIC', lwd=1)
    axis(1, 1:9, tck=-0.02)
    lines(x=bic_our$nmix, y=bic_our$BIC, col='blue2', lwd=2)
    text(7, mean(bic_our$BIC), labels='TOD', cex=4, col="black", font=3)
dev.off()

#~ store/xport the PDFs (and their PARAMETERS) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
TAB_PDFS <- function( LIST, EXES, FACT2, FILO ){# LIST=mv_our ; EXES=play_our ; FACT2=24
# gather total pdfs
    last = lapply(1:length(LIST), function(i) LIST[[i]][[1]])
    nams = paste('mix', seq(1,length(LIST)), sep='')
# gather x's
# https://stackoverflow.com/a/69381173/5885810
    axis = as.vector(EXES$x)
# flip.it and displace.it
    sixa = axis*-1 + 2*pi + pi/2
# https://stackoverflow.com/a/56732068/5885810
    sixa = mapply(function(x) if (x>2*pi) (x-2*pi) else x, sixa)
    # create the whole.df
    pdfs = as_tibble( as.data.frame(last, col.names=nams) ) %>%
        add_column(xR=axis, .before='mix1') %>% add_column(xPY=sixa, .before='xR')
    # base::plot(pdfs$xR, pdfs$mix5)
    # base::plot(pdfs$xPY, pdfs$mix5)
# xport.it
    # write.table(pdfs, file=FILO, sep=',', col.names=T, row.names=F, quote=F)
    write.table(apply(pdfs, 2, function(x) sprintf('%.7f', x)),
        file=FILO, sep=',', col.names=T, row.names=F, quote=F)

# gather MvMF-PDF.parameters and xport them
    parm = rbindlist( lapply(1:length(LIST), function(i)
        as.data.frame( LIST[[i]][[2]][1:3] ) %>%
            add_column(nf=paste('f', seq(1,i), sep=''), .before='mu') %>%
            add_column(nmix=sprintf('mix%s',i), .before='nf') ))
# "tranform" MU into radians (i don't think this is quite OK)
# https://dplyr.tidyverse.org/reference/mutate.html
    parm = parm %>% mutate(muRAD=mu *2*pi / FACT2)
    write.table(parm, file=gsub('.csv','_pars.csv', FILO),
        sep=',', col.names=T, row.names=F, quote=F)
}

TAB_PDFS( mv_doy, play_doy, 365, './model_input/data_WG/pre_processing_circular--DOYmix.csv' )
TAB_PDFS( mv_our, play_our,  24, './model_input/data_WG/pre_processing_circular--TODmix.csv' )

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#- RUN IT ONLY ONCE! (THEN WORK WITH THE OUTPUTS) ---------------------- (END) #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#- PLOTTING ---------------------------------------------------------- (START) #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# read the previously exported pdfs
TOD_PDF = read_csv('./model_input/data_WG/pre_processing_circular--TODmix.csv', col_names=TRUE, comment='#')
DOY_PDF = read_csv('./model_input/data_WG/pre_processing_circular--DOYmix.csv', col_names=TRUE, comment='#')

# https://stat.ethz.ch/pipermail/r-help/2015-May/428989.html
TODx = circular(TOD_PDF$xR, type='angles', units='radians', template='clock24', modulo='2pi', zero=0)
DOYx = circular(DOY_PDF$xR, type='angles', units='radians', template='geographics', modulo='2pi', zero=0)

#~ circular plot for TOD ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# # if doing it right after the computations...
# MIXVMF = lapply(mv_our, '[[', 1)
# PLYSET = play_our
#     lines.circular(PLYSET$x, MIXVMF[[1]]/(SHRNK*SS), lwd=1.5, lty=4, col='#5c3c92', rotation='counter', zero=0, join=F)
# # if not... the above line is replaced by:
#     lines.circular(TODx, TOD_PDF$mix1/(SHRNK*SS), lwd=1.5, lty=4, col='#5c3c92', rotation='counter', zero=0, join=F)

TBIN = 24 *5
TRES = 150
bino = hist(cour, breaks=seq(0, 2*pi, length.out=TBIN+1), plot=F)
new_cont = round(bino$counts /TRES, 0)
new_data = rep(bino$breaks[1:length(bino$breaks)-1] + diff(bino$breaks) /2, new_cont)
CIRC_TOD = circular(new_data, type='angles', units='radians', template='clock24', modulo='2pi', zero=0)

pdf('./plot/pre_processing_circular--plot12_TOD.pdf', width=5, height=5)#, family='Trebuchet MS') 'FiraCR','FiraCSB','russo','b6'
    par(mar=c(0.0,0.0,0.0,0.0), col='white', col.axis='transparent', bg='white', lend=2)
    op <- par(family='russo')
    SHRNK = 1.83
    SS = 0.183
    plot(CIRC_TOD, stack=T, shrink=SHRNK, col='#a7d5ed',#'#a2d5c6',#pch=1,lwd=0.3
        cex=0.89, sep=0.066, start.sep=+0.018, bins=TBIN, ticks=F, tcl.text=+0.13,
        control.circle=circle.control(type='n',col='#044a05', lwd=0.15, lty='2323'))
    ticks.circular(x=circular(seq(0, 2*pi, pi/12)), col='#a5a391', tcl=0.07, lwd=0.4)
    axis.circular(at=circular(seq(0, 2*pi-pi/12, pi/6), units='radians', zero=pi/2, rotation='clock'),
        labels=c('24', seq(2,22,2)), cex=0.75, col='#343837', tcl=0.025, tcl.text=0.15, digits=2, #lty,lwd
        units=NULL, template=NULL, modulo=NULL, zero=NULL, rotation=NULL, tick=F)
    lines.circular(TODx, TOD_PDF$mix1/(SHRNK*SS), lwd=1.3, lty=4, col='#343837', rotation='counter', zero=0, join=F)#'#5c3c92'
    lines.circular(TODx, TOD_PDF$mix5/(SHRNK*SS), lwd=1.9, lty=1, col='#33b864', rotation='counter', zero=0, join=F)
    lines.circular(TODx, TOD_PDF$mix3/(SHRNK*SS), lwd=3.1, lty=1, col='#c23728', rotation='counter', zero=0, join=F)
    rose.diag(cour, bins=24*1, radii.scale='sqrt', axes=F, col=NA, prop=SHRNK*1.19, add=T, border='#63bff0', lwd=0.3, ticks=F)#,lty='2222')
    text(+1.75, +0.93, 'b', cex=2, col='black', font=1)
# https://stackoverflow.com/a/7137354/5885810
    par(family='merri')
# https://stackoverflow.com/a/40505593/5885810
    legend(x=+0.11, y=-1.29, legend=c('1-MvM','5-MvM','3-MvM\n(optimal fit)',sprintf('12min-bins data\n(%d counts/dot)',TRES)),
        pch=c(rep(NaN,3), 19), lty=c(4,1,1,NaN), lwd=c(1.3,1.9,3.1,NaN), text.width=0.55, #tor... ext.width='string_here'
        col=c('#343837', '#33b864', '#c23728', '#a7d5ed'), ncol=2, horiz=FALSE, bg='transparent',#'white'#'ghostwhite'
        bty='o', cex=.85, box.lwd=0.1, box.lty='solid', box.col='transparent',#'white'#'grey50'
        text.col='black', text.font=1, xjust=0, yjust=1, y.intersp=1.5, x.intersp=0.5, seg.len=1
    )
    par( op )
dev.off()

# https://www.heavy.ai/blog/12-color-palettes-for-telling-better-stories-with-your-data

#~ circular plot for DOY ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# BBIN = 365 /2.5
DBIN = 366 /3
DRES = 350
bino = hist(cdoy, breaks=seq(0, 2*pi, length.out=DBIN+1), plot=F)
new_cont = round(bino$counts /DRES, 0)
new_data = rep(bino$breaks[1:length(bino$breaks)-1] + diff(bino$breaks) /2 -diff(bino$breaks)[1], new_cont)
CIRC_DOY = circular(new_data, type='angles', units='radians', template='geographics', modulo='2pi', zero=0)

# https://stackoverflow.com/a/15453778/5885810
# https://www.geeksforgeeks.org/how-to-create-a-range-of-dates-in-r/
dticks = c(365, yday( seq(ymd('2021/02/01'), ymd('2021/12/01'), 'months')))

pdf('./plot/pre_processing_circular--plot11_DOY.pdf', width=5, height=5)#, family='Trebuchet MS') 'FiraCR','FiraCSB','russo','b6'
    par(mar=c(0.0,0.0,0.0,0.0), col='white', col.axis='transparent', bg='white', lend=2)
    # op <- par(family='bauh')
    op <- par(family='russo')
    SHRNK = 1.83
    SS = 0.432
    plot(CIRC_DOY, stack=T, shrink=SHRNK, col='#ffb400',#pch=1,lwd=0.3
        cex=0.89, sep=0.066, start.sep=+0.018, bins=DBIN, ticks=F, tcl.text=+0.13,
        control.circle=circle.control(type='n',col='#044a05', lwd=0.2, lty='2323'))
    ticks.circular(x=circular(dticks /365 *2*pi), col='#a5a391', tcl=0.07, lwd=0.4)
    axis.circular(at=circular(dticks /365 *2*pi, units='radians', zero=pi/2, rotation='clock'),
        labels=dticks, cex=0.75, col='#343837', tcl=0.025, tcl.text=0.17, digits=2,#lty,lwd
        units=NULL, template=NULL, modulo=NULL, zero=NULL, rotation=NULL, tick=F)
    lines.circular(DOYx, DOY_PDF$mix1/(SHRNK*SS), lwd=1.3, lty=4, col='#343837', rotation='counter', zero=0, join=F)
    lines.circular(DOYx, DOY_PDF$mix3/(SHRNK*SS), lwd=1.9, lty=1, col='#33b864', rotation='counter', zero=0, join=F)
    lines.circular(DOYx, DOY_PDF$mix5/(SHRNK*SS), lwd=3.1, lty=1, col='#363445', rotation='counter', zero=0, join=F)
    rose.diag(cdoy, bins=365/5, radii.scale='sqrt', axes=F, col=NA, prop=SHRNK*1.25, add=T, border='#d2980d', lwd=0.3, ticks=F)#,lty='2222')
    text(+1.75, +0.93, 'a', cex=2, col='black', font=1)
# https://stackoverflow.com/a/7137354/5885810
    par(family='merri')
# https://stackoverflow.com/a/40505593/5885810
    legend(x=+0.11, y=-1.29, legend=c('1-MvM','3-MvM','5-MvM\n(optimal fit)',sprintf('3day-bins data\n(%d counts/dot)',DRES)),
        pch=c(rep(NaN,3), 19), lty=c(4,1,1,NaN), lwd=c(1.3,1.9,3.1,NaN), text.width=0.55, #tor... ext.width='string_here'
        col=c('#343837', '#33b864', '#363445', '#ffb400'), ncol=2, horiz=FALSE, bg='transparent',#'white'#'ghostwhite'
        bty='o', cex=.85, box.lwd=0.1, box.lty='solid', box.col='transparent',#'white'#'grey50'
        text.col='black', text.font=1, xjust=0, yjust=1, y.intersp=1.5, x.intersp=0.5, seg.len=1
    )
    par( op )
dev.off()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#- PLOTTING ------------------------------------------------------------ (END) #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#- OUTSOURCING THE CONCATENATING... ---------------------------------- (START) #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# NOT TESTED IN WOS (but should easily work there)

# # https://stackoverflow.com/a/6184547/5885810
# 72 points == 1 inch == 25.4 millimeters
# 5 in -> 5*72 == 360 pts

# # crop them the same amount in all sides... or you'll have problems...
# gs -o pre_processing_circular--plot01_DOYcrop.pdf -sDEVICE=pdfwrite -c "[/CropBox [0 5 360 310]" -c " /PAGES pdfmark" -f pre_processing_circular--plot01_DOY.pdf
# gs -o pre_processing_circular--plot02_TODcrop.pdf -sDEVICE=pdfwrite -c "[/CropBox [0 5 360 310]" -c " /PAGES pdfmark" -f pre_processing_circular--plot02_TOD.pdf

# # alternatively... you can do it in one line with PDFJAM:
# # activate FRAME so you can see what you're doing...
# pdfjam pre_processing_circular--plot1*.pdf --nup 1x2 --frame true --trim '0pt 5pt 5pt 50pt' --papersize '{355pt,610pt}' --noautoscale true --clip true -o pre_processing_circular--plot20_join.pdf
# # note how 360==(0+5)+355 and 720==2*(5+50)+610
pdfjam pre_processing_circular--plot1*.pdf --nup 1x2 --frame false --trim '0pt 5pt 5pt 50pt' --papersize '{355pt,610pt}' --noautoscale true --clip true -o pre_processing_circular--plot20_join.pdf
# https://github.com/rrthomas/pdfjam
# https://tex.stackexchange.com/a/565094
# https://unix.stackexchange.com/a/161721
# https://helpmanual.io/help/pdfjam/
# https://ctan.org/tex-archive/support/pdfjam/#documentation
# https://wiki.uio.no/mn/geo/geoit/index.php/Command_line_tools_to_edit_pdfs

# use GHOSTSCRIPT to chop the top (or some other) border
# # note how 355==355
gs -o pre_processing_circular--plot21_jcut.pdf -sDEVICE=pdfwrite -c "[/CropBox [0 0 355 570]" -c " /PAGES pdfmark" -f pre_processing_circular--plot20_join.pdf
# https://stackoverflow.com/a/6184547/5885810
# https://ghostscript.com/docs/9.54.0/Use.htm
# https://stackoverflow.com/a/26538575/5885810

# transform it into JPG (via PDFTOPPM or IMAGEMAGICK or GHOSTSCRIPT?)...
# this PDFTOPPM is a good option as it doesn't blow up the size
pdftoppm -jpeg -r 200 -singlefile pre_processing_circular--plot21_jcut.pdf pre_processing_circular--plot21_jcut
# convert -verbose -density 150 -trim pre_processing_circular--plot21_jcut.pdf -quality 100 -flatten -sharpen 0x1.0 pre_processing_circular--plot21_jcut.jpg
# convert -density 288 pre_processing_circular--plot21_jcut.pdf -resize 25% pre_processing_circular--plot21_jcut.jpg
# gs -dBATCH -dNOPAUSE -dSAFER -sDEVICE=jpeg -dJPEGQ=95 -r300x300 -sOutputFile=pre_processing_circular--plot21_jcut.jpg pre_processing_circular--plot21_jcut.pdf
#
# https://stackoverflow.com/a/61700520/5885810
# https://docs.oracle.com/cd/E88353_01/html/E37839/pdftoppm-1.html
# https://superuser.com/a/168474
# https://legacy.imagemagick.org/discourse-server/viewtopic.php?t=30644

# some more chopping...
convert pre_processing_circular--plot21_jcut.jpg -gravity North -chop 0x90 pre_processing_circular--plot21_jcut.jpg
# https://superuser.com/a/1161341

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#- OUTSOURCING THE CONCATENATING... ------------------------------------ (END) #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
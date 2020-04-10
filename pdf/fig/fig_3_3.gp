set terminal qt size 640, 480 position 1000, 300
# set terminal qt size 640, 480 position 280, 300

# set terminal postscript eps color size 5, 3.5 colortext font "Helvetica, 14"
# set terminal gif size 640, 480 font "Helvetica, 12" animate delay 10
# set terminal png truecolor font "Helvetica, 12" size 640, 480

# set table ""
FILENAME = "fig_3_3.eps"
# set output FILENAME

# set title ""
set xlabel "y_m"
set ylabel "E"
# set format x "%.0f"
# set format y "%.0f"

unset key
set grid
# set size square

# set logscale x
# set logscale y

set xrange [0:1]
set yrange [0:6]

# set timestamp "%Y/%m/%d (%a) %H:%M:%S"

set pointsize 1.0
set samples 300

# set multiplot layout 2, 2

plot -log(x)












# unset multiplot
pause -1 "Press any key: "


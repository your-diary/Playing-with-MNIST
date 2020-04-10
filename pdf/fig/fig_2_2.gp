set terminal qt size 640, 480 position 1000, 300
# set terminal qt size 640, 480 position 280, 300

# set terminal postscript eps color size 5, 3.5 colortext font "Helvetica, 14"
# set terminal gif size 640, 480 font "Helvetica, 12" animate delay 10
# set terminal png truecolor font "Helvetica, 12" size 640, 480

# set table ""
FILENAME = "004.eps"
# set output FILENAME

# set title ""
# set xlabel ""
# set ylabel ""
# set format x "%.0f"
# set format y "%.0f"

# unset key
# set grid
# set size square

# set logscale x
# set logscale y

set xrange [0:1.2]
set yrange [0:1.2]

# set timestamp "%Y/%m/%d (%a) %H:%M:%S"

set pointsize 1.0
# set samples 200

# set multiplot layout 2, 2

$DATA << EOD
0 0
0 1
1 0
1 1
EOD

set key width 1
set xlabel "x_1"
set ylabel "x_2"
set pointsize 1.5
set size square

#0.5x + 0.5y = 0.7
#-0.5x - 0.5y = -0.7
#0.5x + 0.5y = 0.4

plot $DATA notitle,\
    (0.7 - 0.5 * x) / 0.5 title "AND",\
    (-0.7 + 0.5 * x) / (-0.5) title "NAND",\
    (0.4 - 0.5 * x) / (0.5) title "OR"

# unset multiplot
pause -1 "Press any key: "

